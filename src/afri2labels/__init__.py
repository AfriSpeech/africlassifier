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

__all__ = [
    "Afri2LabelsClassifier",
    "SpeechTopicClassifier",
    "PredictionResult",
    "LabelScore",
    "extract_concept_labels",
    "ISO_TO_NAME",
    "NAME_TO_ISO",
    "resolve_language",
]
