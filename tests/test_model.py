"""Unit and behavioral tests for the sentiment model."""

import pytest

from app.config import ALLOWED_LABELS
from app.model import SentimentModel


def test_preprocess_normalizes_mentions_and_urls() -> None:
    text = "Thanks @machineinnovators see https://example.com"
    result = SentimentModel.preprocess(text)
    assert result == "Thanks @user see http"


def test_allowed_labels_are_expected_three_classes() -> None:
    assert ALLOWED_LABELS == {"negative", "neutral", "positive"}


@pytest.fixture(scope="module")
def real_model() -> SentimentModel:
    """Load the pretrained model once for behavioral checks."""
    return SentimentModel()


def test_model_returns_allowed_label(real_model: SentimentModel) -> None:
    result = real_model.predict("The product arrived yesterday.")

    assert result.sentiment in ALLOWED_LABELS
    assert 0.0 <= result.confidence <= 1.0


def test_model_detects_clearly_positive_text(real_model: SentimentModel) -> None:
    result = real_model.predict("I absolutely love this product. It is fantastic!")

    assert result.sentiment == "positive"


def test_model_detects_clearly_negative_text(real_model: SentimentModel) -> None:
    result = real_model.predict("This product is terrible. I hate it.")

    assert result.sentiment == "negative"
