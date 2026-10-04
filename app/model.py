"""Sentiment model loading and inference logic."""

from dataclasses import dataclass

from transformers import pipeline

from app.config import ALLOWED_LABELS, MODEL_NAME, MODEL_VERSION


@dataclass(frozen=True)
class SentimentResult:
    """Normalized inference result."""

    sentiment: str
    confidence: float
    model_version: str


class SentimentModel:
    """Wrapper around the pretrained CardiffNLP sentiment model.

    The Hugging Face pipeline is created once when this class is instantiated.
    The FastAPI lifespan creates a single instance at application startup, so
    the model is not reloaded for every request.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._classifier = pipeline(
            task="text-classification",
            model=model_name,
            tokenizer=model_name,
        )

    @staticmethod
    def preprocess(text: str) -> str:
        """Apply the preprocessing recommended by the model card.

        User mentions are normalized to ``@user`` and links to ``http``.
        """
        normalized_tokens: list[str] = []
        for token in text.split():
            if token.startswith("@") and len(token) > 1:
                token = "@user"
            elif token.startswith("http"):
                token = "http"
            normalized_tokens.append(token)
        return " ".join(normalized_tokens)

    def predict(self, text: str) -> SentimentResult:
        """Return one of: negative, neutral, positive."""
        processed_text = self.preprocess(text)
        prediction = self._classifier(processed_text, truncation=True)[0]

        label = str(prediction["label"]).lower()
        if label not in ALLOWED_LABELS:
            raise RuntimeError(f"Unexpected model label: {label}")

        return SentimentResult(
            sentiment=label,
            confidence=float(prediction["score"]),
            model_version=MODEL_VERSION,
        )
