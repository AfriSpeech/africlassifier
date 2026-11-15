"""speech2topics: Multimodal EmbeddingGemma 2 for African Speech to Concepts."""

from speech2topics.classifier import (
    Afri2LabelsClassifier,
    LabelScore,
    PredictionResult,
    SpeechTopicClassifier,
)
from speech2topics.extractor import extract_concept_labels
from speech2topics.languages import ISO_TO_NAME, NAME_TO_ISO, resolve_language

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
