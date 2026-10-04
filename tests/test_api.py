"""Contract tests for the FastAPI inference service."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import ALLOWED_LABELS, MAX_TEXT_LENGTH, MODEL_VERSION
from app.model import SentimentResult


class FakeSentimentModel:
    """Small deterministic substitute used by API tests."""

    def predict(self, text: str) -> SentimentResult:
        sentiment = "negative" if "bad" in text.lower() else "positive"
        return SentimentResult(
            sentiment=sentiment,
            confidence=0.99,
            model_version=MODEL_VERSION,
        )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Start the API without downloading the Hugging Face model."""
    monkeypatch.setattr(main_module, "SentimentModel", FakeSentimentModel)
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_endpoint_reports_ready_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
    }


def test_predict_returns_expected_contract(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"text": "I really love this new product!"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"sentiment", "confidence", "model_version"}
    assert body["sentiment"] in ALLOWED_LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == MODEL_VERSION


def test_predict_rejects_empty_text(client: TestClient) -> None:
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_predict_rejects_whitespace_only_text(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422


def test_predict_rejects_text_over_max_length(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"text": "x" * (MAX_TEXT_LENGTH + 1)},
    )

    assert response.status_code == 422
