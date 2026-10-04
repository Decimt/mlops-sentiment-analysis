"""Prometheus metrics used by the sentiment inference service."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "sentiment_requests_total",
    "Total number of HTTP requests received by the service.",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "sentiment_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

PREDICTION_COUNT = Counter(
    "sentiment_predictions_total",
    "Total number of sentiment predictions by predicted class.",
    ["sentiment"],
)

INPUT_LENGTH = Histogram(
    "sentiment_input_length_chars",
    "Input text length in characters for prediction requests.",
    buckets=(25, 50, 100, 200, 300, 400, 512),
)
