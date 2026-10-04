# Architecture

## Purpose

This document describes the architecture of the MLOps sentiment-analysis project
at component and flow level.

The repository README provides the operational overview; this document focuses
on responsibilities, boundaries, and data/control flow.

---

## High-level component model

```text
                        ┌────────────────────┐
                        │      GitHub        │
                        └─────────┬──────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │  GitHub Actions    │
                        │ CI + image build   │
                        └─────────┬──────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │       GHCR         │
                        └────────────────────┘


        request
Client ──────────────────────────────────────────────┐
                                                    ▼
                                           ┌────────────────┐
                                           │    FastAPI     │
                                           ├────────────────┤
                                           │ /health        │
                                           │ /predict       │
                                           │ /metrics       │
                                           └───────┬────────┘
                                                   │
                                                   ▼
                                         ┌───────────────────┐
                                         │ CardiffNLP model  │
                                         │ + tokenizer       │
                                         └───────────────────┘

                                                   │
                                                   │ metrics
                                                   ▼
                                           ┌────────────────┐
                                           │   Prometheus   │
                                           └───────┬────────┘
                                                   │
                                      ┌────────────┴────────────┐
                                      ▼                         ▼
                                ┌───────────┐             Alert rules
                                │  Grafana  │
                                └───────────┘


Periodic control plane
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Apache Airflow                          │
│ collect → validate → evaluate/drift → branch → retrain/skip   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Inference plane

The inference plane is intentionally small.

### FastAPI

FastAPI is responsible for:

- request validation;
- HTTP contracts;
- service health;
- invoking the model wrapper;
- exposing metrics.

The API does not own orchestration logic or long-running training jobs.

### Model wrapper

The model wrapper encapsulates the Hugging Face sentiment pipeline.

The model is initialized once during the FastAPI application lifecycle. This is
important because model initialization is significantly more expensive than a
single inference call.

The wrapper also keeps model-specific behavior outside the HTTP route layer.

### Pydantic schemas

Schemas provide explicit contracts for:

- prediction request;
- prediction response;
- health response.

This prevents ad hoc payloads from leaking into model code.

---

## Observability plane

### Application metrics

The API provides operational and ML-oriented metrics.

Operational:

- request count;
- status-code distribution;
- request latency.

ML-oriented:

- predicted sentiment distribution;
- input-length distribution.

The monitoring endpoints `/metrics` and `/health` are excluded from custom
request-volume and latency metrics so internal health/scrape traffic does not
distort user-facing traffic indicators.

### Prometheus

Prometheus periodically scrapes `/metrics`.

Responsibilities:

- metric collection;
- time-series storage;
- PromQL evaluation;
- alert-rule evaluation.

### Grafana

Grafana reads Prometheus as its data source.

The dashboard presents:

- request rate;
- HTTP error rate;
- P95 latency;
- sentiment distribution;
- average input length.

---

## Delivery plane

### GitHub Actions

The workflow separates quality validation from container delivery.

```text
push / pull request
        │
        ▼
quality job
├── Ruff
├── Black
└── pytest
        │
        ▼
container job
        │
        ├── pull request → build only
        │
        └── main push    → build + publish
```

The container stage depends on successful quality validation.

### GitHub Container Registry

For successful pushes to `main`, the workflow publishes:

- `latest`;
- a commit-specific tag.

The commit tag provides source-to-artifact traceability.

---

## Orchestration plane

Apache Airflow is isolated from the FastAPI runtime.

This is an intentional boundary:

- FastAPI handles online inference;
- Airflow handles scheduled lifecycle orchestration.

The DAG models:

1. data collection;
2. data validation;
3. current-model evaluation;
4. drift assessment;
5. retraining decision;
6. retraining or skip branch;
7. candidate evaluation;
8. candidate registration;
9. pipeline completion.

The current tasks use lightweight simulated values to demonstrate dependency,
branching, and information flow.

A real implementation would replace them with calls to data platforms,
evaluation jobs, training infrastructure, registries, and deployment systems.

---

## Drift architecture

Two categories are treated differently.

### Data drift

Data drift concerns changes in `P(X)`.

Signals available or proposed include:

- text-length distribution;
- predicted-class distribution;
- language distribution;
- token/hashtag/URL/mention statistics;
- embedding-distribution comparison.

A distribution shift is an investigation signal, not proof of failure.

### Concept drift

Concept drift concerns changes in `P(Y|X)`.

Reliable detection requires a source of truth, such as:

- delayed business outcomes;
- human labeling;
- curated evaluation samples.

The architecture therefore treats concept-drift detection as a feedback loop,
not as something inferred from prediction counts alone.

---

## Runtime topology

The local environment uses Docker Compose because it is sufficient for the
number of services involved and keeps the project reproducible without adding
Kubernetes complexity.

The stack contains separate runtime components for:

- API;
- Prometheus;
- Grafana;
- Airflow.

Container networking allows internal service-to-service communication while
selected ports are exposed to the local host.

---

## Design principles

The project follows a few simple principles:

1. **Keep inference and orchestration separate.**
2. **Load expensive ML artifacts once.**
3. **Test software contracts separately from model behavior.**
4. **Fail CI before publishing artifacts.**
5. **Treat observability as part of the application design.**
6. **Treat drift signals as evidence to investigate, not automatic proof.**
7. **Keep the educational implementation small enough to remain runnable.**
