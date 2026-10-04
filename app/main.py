"""FastAPI entrypoint for the sentiment inference service."""

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.config import MODEL_VERSION
from app.metrics import INPUT_LENGTH, PREDICTION_COUNT, REQUEST_COUNT, REQUEST_LATENCY
from app.model import SentimentModel
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once when the API starts."""
    app.state.sentiment_model = SentimentModel()
    yield
    app.state.sentiment_model = None


app = FastAPI(
    title="MLOps Sentiment Analysis API",
    description=(
        "Inference service based on CardiffNLP's "
        "twitter-roberta-base-sentiment-latest model."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    """Measure request volume, status codes, and latency."""
    start_time = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        endpoint = request.url.path
        elapsed = perf_counter() - start_time

        if endpoint not in {"/metrics", "/health"}:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()

            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(elapsed)


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health(request: Request) -> HealthResponse:
    """Return service and model readiness information."""
    model_loaded = getattr(request.app.state, "sentiment_model", None) is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=MODEL_VERSION,
    )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Classify the input text as negative, neutral, or positive."""
    model: SentimentModel | None = getattr(request.app.state, "sentiment_model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    INPUT_LENGTH.observe(len(payload.text))
    result = model.predict(payload.text)
    PREDICTION_COUNT.labels(sentiment=result.sentiment).inc()

    return PredictionResponse(
        sentiment=result.sentiment,
        confidence=result.confidence,
        model_version=result.model_version,
    )
