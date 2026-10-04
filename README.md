# MLOps Sentiment Analysis

End-to-end educational MLOps project for social-media sentiment classification.

The project exposes the pretrained Hugging Face model
`cardiffnlp/twitter-roberta-base-sentiment-latest` through a FastAPI service and
builds the surrounding MLOps lifecycle: automated testing, code-quality gates,
containerization, CI/CD, observability, alerting, drift-oriented monitoring, and
periodic orchestration with Apache Airflow.

The focus is not model training quality itself, but the reproducibility,
operability, traceability, and maintainability of the system around the model.

---

## Project objectives

The original business problem is a manual social-media sentiment monitoring
process that is slow, inconsistent, and unable to detect changes in public
perception quickly.

This project turns a pretrained sentiment model into a small production-like
service that can:

- classify text as `negative`, `neutral`, or `positive`;
- expose predictions through a REST API;
- validate inputs and return a stable response contract;
- run automated API and model-behavior tests;
- enforce linting and formatting standards;
- package the service as a Docker image;
- run CI/CD automatically through GitHub Actions;
- publish versioned container images to GitHub Container Registry;
- expose Prometheus-compatible application and ML-oriented metrics;
- visualize service behavior through Grafana;
- evaluate an operational alert;
- model periodic evaluation, drift checks, and retraining decisions in Airflow.

---

## System architecture

```text
                              GitHub
                                │
                                ▼
                     GitHub Actions CI/CD
                    ┌───────────┴───────────┐
                    │                       │
             Quality Gate              Container Build
          Ruff / Black / pytest              │
                    │                        ▼
                    └──────────────────────► GHCR


Client / Consumer
       │
       ▼
 ┌───────────────┐
 │    FastAPI    │
 │               │
 │ GET /health   │
 │ POST /predict │
 │ GET /metrics  │
 └───────┬───────┘
         │
         ▼
 ┌─────────────────────────────────────────────┐
 │ cardiffnlp/twitter-roberta-base-            │
 │ sentiment-latest                            │
 │ negative / neutral / positive               │
 └─────────────────────────────────────────────┘

         │
         │ metrics
         ▼
   ┌────────────┐
   │ Prometheus │
   └─────┬──────┘
         │
         ▼
    ┌─────────┐
    │ Grafana │
    └─────────┘
         │
         └── operational alert


                  Periodic MLOps lifecycle
                            │
                            ▼
                     Apache Airflow
                            │
                            ▼
                    collect_new_data
                            │
                            ▼
                      validate_data
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          evaluate_current_model   detect_drift
                  └─────────┬─────────┘
                            ▼
                    decide_retraining
                     /             \
                    ▼               ▼
             retrain_model     skip_retraining
                    │               │
                    ▼               │
          evaluate_candidate       │
                    │               │
                    ▼               │
          register_candidate       │
                    └───────┬───────┘
                            ▼
                     pipeline_complete
```

Detailed architecture notes are available in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Technology stack

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| API | FastAPI |
| Validation | Pydantic |
| ML inference | Hugging Face Transformers + PyTorch |
| Model | CardiffNLP `twitter-roberta-base-sentiment-latest` |
| Testing | pytest |
| Linting | Ruff |
| Formatting | Black |
| Containerization | Docker |
| Multi-service runtime | Docker Compose |
| CI/CD | GitHub Actions |
| Container registry | GitHub Container Registry (GHCR) |
| Metrics | Prometheus client |
| Monitoring | Prometheus |
| Visualization | Grafana |
| Orchestration | Apache Airflow |

---

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── airflow/
│   └── dags/
│       └── sentiment_mlops_pipeline.py
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── metrics.py
│   ├── model.py
│   └── schemas.py
│
├── docs/
│   ├── ARCHITECTURE.md
│
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── provisioning/
│   └── prometheus/
│       ├── alerts.yml
│       └── prometheus.yml
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

---

## Inference service

The FastAPI application loads the model once during application startup rather
than on every request.

### `GET /health`

Returns service readiness and model information.

Example:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

### `POST /predict`

Example request:

```json
{
  "text": "I really love this product!"
}
```

Example response:

```json
{
  "sentiment": "positive",
  "confidence": 0.9844,
  "model_version": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

The API validates the input with Pydantic and rejects invalid requests such as
empty or overly long text.

### `GET /metrics`

Exposes Prometheus-compatible metrics.

Internal monitoring endpoints such as `/metrics` and `/health` are excluded from
the custom application request counters so that monitoring traffic does not
inflate service-usage metrics.

---

## Local setup

Python 3.11 is the reference runtime.

### 1. Create the virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 2. Run the API locally

```powershell
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

---

## Testing and code quality

The local quality gate is:

```powershell
ruff check .
black --check .
pytest
```

The test strategy deliberately separates two concerns.

### API contract tests

API tests use a lightweight fake model where appropriate so HTTP behavior can
be tested quickly and deterministically.

They cover areas such as:

- health endpoint behavior;
- prediction response schema;
- accepted labels;
- invalid input;
- empty input;
- excessive input length.

### Model behavioral tests

A smaller group of tests uses the real pretrained CardiffNLP model.

These tests verify:

- prediction labels belong to the expected class set;
- clearly positive examples are classified as positive;
- clearly negative examples are classified as negative.

The tests do not assert an exact floating-point confidence score because that
would create a brittle contract around implementation details.

---

## Containerization

The application image is built from `python:3.11-slim`.

The Dockerfile copies dependency definitions before application code so Docker
can reuse cached dependency layers when only source files change.

Build manually:

```powershell
docker build -t mlops-sentiment-api:local .
```

Run:

```powershell
docker run --rm `
  -p 8000:8000 `
  --name sentiment-api `
  mlops-sentiment-api:local
```

For this educational implementation, the Hugging Face model is downloaded when
a fresh runtime first needs it. A production system could instead pin and
pre-package model artifacts or retrieve them from a model registry.

---

## CI/CD

The GitHub Actions workflow is defined in:

```text
.github/workflows/ci-cd.yml
```

### Continuous Integration

For pushes and pull requests, the quality job:

1. checks out the repository;
2. configures Python 3.11;
3. installs dependencies;
4. runs `ruff check .`;
5. runs `black --check .`;
6. runs `pytest`.

The container job depends on the quality job. A failing quality gate prevents a
container artifact from being produced or published.

### Container build and publication

The workflow builds the Docker image after successful CI.

For pushes to `main`, the image is published to GitHub Container Registry with:

- a convenient `latest` tag;
- a tag derived from the exact Git commit SHA.

The SHA tag provides traceability between source code and container artifact.

A public hosted deployment is intentionally not required by this project. A
future CD stage could deploy the validated image to a managed runtime such as a
Hugging Face Docker Space or another container platform.

---

## Monitoring and observability

The complete monitoring stack is started with Docker Compose.

```powershell
docker compose up --build
```

Local services:

| Service | Address |
|---|---|
| FastAPI | `http://localhost:8000` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |
| Airflow | `http://localhost:8080` |

Demo credentials:

```text
Grafana
username: admin
password: admin

Airflow
username: admin
password: admin
```

These credentials are intentionally simple because the stack is local and
educational. They must not be used in a real deployment.

### Application metrics

The API exposes:

- `sentiment_requests_total`
- `sentiment_request_latency_seconds`
- `sentiment_predictions_total`
- `sentiment_input_length_chars`

These provide both operational and ML-oriented signals.

### Grafana dashboard

The automatically provisioned dashboard shows:

- prediction request rate;
- HTTP error rate;
- P95 request latency;
- predicted sentiment distribution;
- average input length.

### Alert

Prometheus evaluates the `HighHttpErrorRate` alert.

The demonstrative rule considers an HTTP 5xx rate above 5% for at least five
minutes anomalous.

The threshold is not universal. In production it should be derived from traffic
volume, service criticality, and formal SLOs.

---

## Drift monitoring strategy

The current project provides signals that can support drift investigation but
does not claim that every distribution change is model degradation.

### Data drift

Potential signals include changes in:

- input length distribution;
- predicted sentiment distribution;
- language distribution;
- token usage;
- hashtags, URLs, and mentions;
- embedding distributions.

The current implementation directly exposes input-length and prediction-class
metrics. Richer text-distribution checks are described as production
extensions.

A shift in prediction distribution is not automatically a failure. For
example, a real reputational event could legitimately increase negative
sentiment.

### Concept drift

Concept drift concerns changes in the relationship between input text and its
true sentiment.

Prediction statistics alone are insufficient to detect it reliably.

A production process would periodically:

1. sample production texts;
2. obtain delayed ground-truth labels or human annotations;
3. compare predictions with those labels;
4. monitor macro-F1, per-class recall, calibration, and other quality metrics;
5. trigger investigation or retraining when quality falls below agreed
   thresholds.

---

## Airflow orchestration

Airflow runs independently from the FastAPI Python environment in its own
container.

The DAG:

```text
sentiment_mlops_lifecycle
```

runs on a weekly schedule:

```text
0 2 * * 0
```

Sunday at 02:00.

Its lifecycle is:

```text
collect_new_data
        ↓
validate_data
        ↓
 ┌──────┴─────────┐
 ▼                ▼
evaluate_model   detect_drift
 └──────┬─────────┘
        ↓
decide_retraining
      /       \
     ▼         ▼
retrain       skip
   ↓
evaluate_candidate
   ↓
register_candidate
   ↓
pipeline_complete
```

The project intentionally uses lightweight simulated values inside the DAG so
the orchestration logic can be demonstrated without turning the exercise into a
full model-training platform.

The demonstrative retraining rule is:

```text
data drift score > 0.20
OR
current macro-F1 < 0.80
```

The thresholds illustrate the control flow and are not presented as universal
production defaults.

In a real system these tasks would call data-ingestion jobs, validation
services, evaluation pipelines, training jobs, model registries, approval
gates, and deployment workflows.

---

## Running the complete stack

From the repository root:

```powershell
docker compose down
docker compose up --build
```

After the services become healthy:

```text
FastAPI     http://localhost:8000
Swagger     http://localhost:8000/docs
Prometheus  http://localhost:9090
Grafana     http://localhost:3000
Airflow     http://localhost:8080
```

To verify prediction metrics, generate a request:

```powershell
$body = @{
    text = "I really love this product!"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/predict" `
    -ContentType "application/json" `
    -Body $body
```

Then inspect:

```powershell
curl.exe http://127.0.0.1:8000/metrics
```

---

## Design choices

### Why a pretrained model?

The project brief focuses on MLOps rather than model research. Using an existing
pretrained model keeps attention on serving, reproducibility, deployment,
monitoring, and orchestration.

### Why load the model once?

Model initialization is expensive compared with inference. Loading it during
application startup avoids repeated initialization on every request.

### Why separate API tests from model tests?

HTTP contracts should remain fast and deterministic, whereas real model
behavior is slower and depends on the ML runtime. Separating them keeps the
test suite understandable and reduces unnecessary coupling.

### Why Docker Compose?

The project contains multiple cooperating components but does not require a
full Kubernetes platform. Docker Compose is sufficient to demonstrate service
separation, networking, configuration, and repeatable local startup.

### Why separate Airflow from the API environment?

Airflow has a large dependency surface and a different responsibility from the
inference service. Running it in its own container avoids dependency conflicts
and preserves separation of concerns.

---

## Project delivery

The GitHub repository contains the complete executable project.

The final Google Colab notebook is used as the review document and describes:

- the incremental development path;
- architectural choices;
- test strategy;
- CI/CD behavior;
- monitoring and alert thresholds;
- drift considerations;
- Airflow orchestration;
- screenshots and validation evidence;
- limitations and future improvements.

The Colab link can be added here once the final notebook is published.

---

## License and usage

This repository is an educational MLOps project. The underlying pretrained model
is provided by CardiffNLP through Hugging Face and remains subject to its own
model-card and licensing terms.
