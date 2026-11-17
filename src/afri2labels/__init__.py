"""speech2topics: Multimodal EmbeddingGemma 2 for African Speech to Concepts."""

"""afri2labels: Multimodal Foundation Model mapping African Speech Audio & Text to Concepts."""

from afri2labels.classifier import (
    Afri2LabelsClassifier,
    LabelScore,
    PredictionResult,
    SpeechTopicClassifier,
)
from afri2labels.extractor import extract_concept_labels
from afri2labels.languages import ISO_TO_NAME, NAME_TO_ISO, resolve_language


def create_app(*args, **kwargs):
    """Lazy import of the FastAPI serving application."""
    from afri2labels.server import create_app as _create_app
    return _create_app(*args, **kwargs)


__all__ = [
    "Afri2LabelsClassifier",
    "SpeechTopicClassifier",
    "PredictionResult",
    "LabelScore",
    "extract_concept_labels",
    "ISO_TO_NAME",
    "NAME_TO_ISO",
    "resolve_language",
    "create_app",
]
