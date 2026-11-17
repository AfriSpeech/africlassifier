"""africlassifier: CPU-first Multi-Task Foundation Model for African Speech and Text."""

from africlassifier.classifier import (
    AfriClassifier,
    MultiTaskResult,
    ScoredClass,
    TaskPrediction,
)
from africlassifier.languages import ISO_TO_NAME, NAME_TO_ISO, resolve_language


def create_app(*args, **kwargs):
    """Lazy import of FastAPI serving application."""
    from africlassifier.server import create_app as _create_app
    return _create_app(*args, **kwargs)


__all__ = [
    "AfriClassifier",
    "MultiTaskResult",
    "ScoredClass",
    "TaskPrediction",
    "ISO_TO_NAME",
    "NAME_TO_ISO",
    "resolve_language",
    "create_app",
]
