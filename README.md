# MLOps Sentiment Analysis

Educational MLOps project that exposes the pretrained Hugging Face model
`cardiffnlp/twitter-roberta-base-sentiment-latest` through FastAPI and builds an
incremental path from local inference to testing, containerization and CI/CD.

## Current scope — Phase 4

Implemented:

- FastAPI inference service
- `GET /health`
- `POST /predict`
- model loaded once during application startup
- input validation with Pydantic
- pytest API contract tests
- behavioral tests against the real pretrained model
- Ruff linting
- Black formatting checks
- Docker image based on `python:3.11-slim`
- GitHub Actions CI/CD pipeline
- automatic Docker build after quality checks
- automatic publication to GitHub Container Registry (GHCR) on pushes to `main`

Not implemented yet:

- Prometheus/Grafana monitoring
- alerting
- Airflow orchestration
- production hosting of the running API

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── model.py
│   └── schemas.py
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Local setup

Python 3.11 is the reference runtime.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Run the application:

```powershell
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Quality gate

Run locally before committing:

```powershell
ruff check .
black --check .
pytest
```

The API tests use a fake model to keep HTTP contract tests fast and deterministic.
The behavioral model tests load the real CardiffNLP model once per pytest module
and check valid labels plus clearly positive and clearly negative examples.

## Docker

Build:

```powershell
docker build -t mlops-sentiment-api:phase4 .
```

Run:

```powershell
docker run --rm -p 8000:8000 --name sentiment-api mlops-sentiment-api:phase4
```

Test health:

```powershell
curl.exe http://127.0.0.1:8000/health
```

The model is downloaded when the container starts for the first time. This keeps
the Dockerfile simple for the educational project. A production alternative
would pre-download/pin model artifacts during the image build or use a managed
model registry/cache.

## GitHub Actions CI/CD

Workflow: `.github/workflows/ci-cd.yml`.

### CI job — `quality`

Runs on every pull request to `main` and every push to `main`:

1. checkout
2. Python 3.11 setup
3. pip dependency caching
4. Hugging Face model cache
5. dependency installation
6. `ruff check .`
7. `black --check .`
8. `pytest`

The container job depends on this job, therefore an image is never built or
published when the quality gate fails.

### CD/build job — `container`

For pull requests it builds the Docker image without publishing it.

For pushes to `main`, after CI succeeds, it:

1. authenticates to `ghcr.io` using GitHub's automatically provided
   `GITHUB_TOKEN`
2. builds the Docker image
3. publishes two tags to GitHub Container Registry:
   - `latest`
   - the exact Git commit SHA

This provides a mutable convenience tag and an immutable traceable artifact.
No manually stored registry password is required.

### Deployment choice

Phase 4 publishes the deployable image to GHCR but does not host the running API
on a public production endpoint. The project brief considers a Hugging Face
deployment optional. A future deployment could use a Hugging Face Space with a
Docker SDK: the repository/Space would receive the application files and
Dockerfile, while an HF token would be stored only as a GitHub repository secret
and used by a dedicated deploy step after successful CI/build.

## Model endpoint examples

Health response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

Prediction request:

```json
{
  "text": "I really love this product!"
}
```

Example prediction response:

```json
{
  "sentiment": "positive",
  "confidence": 0.98,
  "model_version": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

## Current limitations

- The pretrained model is used as-is; no fine-tuning is performed yet.
- The model is English-oriented.
- A low-confidence prediction can still be semantically debatable; confidence is
  not a guarantee of correctness.
- The Docker image is large because PyTorch and Transformer dependencies are
  included.
- The current CD publishes a container artifact, but does not yet deploy a
  continuously hosted service.
- Drift monitoring and retraining orchestration belong to later phases.

# Phase 5 - Monitoring

This patch adds the monitoring layer required by the project brief.

## Stack

- FastAPI exposes `/metrics` in Prometheus format.
- Prometheus scrapes the API every 5 seconds.
- Grafana is automatically provisioned with a Prometheus data source and dashboard.
- Prometheus evaluates a `HighHttpErrorRate` alert rule.

## Metrics

- `sentiment_requests_total`: HTTP volume and status codes.
- `sentiment_request_latency_seconds`: request latency histogram.
- `sentiment_predictions_total`: predicted sentiment class distribution.
- `sentiment_input_length_chars`: input text length histogram.

## Start the stack

```bash
docker compose up --build
```

Services:

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Grafana demo credentials:

- username: `admin`
- password: `admin`

The dashboard is provisioned automatically in the `MLOps Sentiment` folder.

## Alert

`HighHttpErrorRate` becomes firing when more than 5% of HTTP requests return a
5xx status for at least five minutes. The threshold is deliberately simple for
this educational project. In production it should be derived from SLOs,
expected traffic, and service criticality.

The rule can be inspected in Prometheus under **Alerts**.

## Drift signals

The monitoring stack provides operational and input-distribution signals, but
these must not be confused with definitive drift detection.

Potential **data drift** signals include changes in:

- average and distribution of input length;
- predicted sentiment class distribution;
- language, tokens, hashtags, URLs, mentions, and embeddings (future work).

A sharp shift is a signal to investigate, not proof of model degradation. For
example, a real reputational crisis could legitimately increase the share of
negative predictions.

**Concept drift** requires labelled production observations or another source
of ground truth. A production workflow would periodically sample incoming
texts, obtain human labels, compare predictions with those labels, and monitor
metrics such as macro-F1, per-class recall, and calibration over time.