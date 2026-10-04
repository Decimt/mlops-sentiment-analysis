"""Pydantic schemas exposed by the API."""

from pydantic import BaseModel, Field, field_validator

from app.config import MAX_TEXT_LENGTH


class PredictionRequest(BaseModel):
    """Request body for sentiment prediction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="English social-media text to classify.",
        examples=["I really love this new product!"],
    )

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only payloads and normalize outer whitespace."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized


class PredictionResponse(BaseModel):
    """Prediction returned by the inference service."""

    sentiment: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    model_loaded: bool
    model_version: str
