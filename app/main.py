"""FastAPI entrypoint for the sentiment inference service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from app.config import MODEL_VERSION
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
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health(request: Request) -> HealthResponse:
    """Return service and model readiness information."""
    model_loaded = getattr(request.app.state, "sentiment_model", None) is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=MODEL_VERSION,
    )


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

    result = model.predict(payload.text)
    return PredictionResponse(
        sentiment=result.sentiment,
        confidence=result.confidence,
        model_version=result.model_version,
    )
