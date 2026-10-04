"""Application configuration."""

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_VERSION = MODEL_NAME
MAX_TEXT_LENGTH = 1000
ALLOWED_LABELS = {"negative", "neutral", "positive"}
