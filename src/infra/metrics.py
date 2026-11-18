"""
custom Prometheus metrics for the AI Detector
Prometheus metrics must be module-level singletons. If they're defined inside a
class or function, duplicate registration errors occur on reload. Keeping them
here makes them importable from anywhere without side effects
"""

from prometheus_client import Counter, Histogram

from src.core.enums import PredictionLabel

# -- Prediction tracking ------------------
PREDICTION_COUNTER = Counter(
    name="ai_detector_predictions_total",
    documentation="Total predictions made, labelled by outcome.",
    labelnames=["label"],
    # Labels: REAL | AI_GENERATED | UNCERTAIN_LEANING_AI |
    #         UNCERTAIN_LEANING_REAL | UNCERTAIN_NEUTRAL
)

# -- Cache behaviour -----------------------
CACHE_COUNTER = Counter(
    name="ai_detector_cache_total",
    documentation="Cache hits and misses for pHash-based prediction caching.",
    labelnames=["result"],
    # Labels: hit | miss
)

# -- Inference latency ----------------------
INFERENCE_HISTOGRAM = Histogram(
    name="ai_detector_inference_duration_seconds",
    documentation="Time spent running ONNX inference (excludes cache hits).",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0],
)

# -- Explain (GradCAM) latency ------------------
EXPLAIN_HISTOGRAM = Histogram(
    name="ai_detector_explain_duration_seconds",
    documentation="Time spent running GradCAM explanation",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0],
)


# -- Start every label series at 0 ------------------
# A counter that first appears already at 1 makes rate() and increase() miss that first event,
# and the donut chart only lists labels that have occurred. Creating the series up front fixes both.
for _label in PredictionLabel:
    PREDICTION_COUNTER.labels(label=_label.value)
for _result in ("hit", "miss"):
    CACHE_COUNTER.labels(result=_result)
