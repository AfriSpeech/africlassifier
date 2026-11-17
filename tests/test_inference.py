"""Unit tests for africlassifier multi-task classification engine."""

import pytest
from africlassifier.languages import ISO_TO_NAME, NAME_TO_ISO, resolve_language
from africlassifier.taxonomies import DOMAINS, EMOTIONS, INTENTS, LANGUAGES, SENTIMENTS


def test_language_resolution():
    iso, name = resolve_language("twi")
    assert iso == "twi"
    assert name == "Twi"

    iso, name = resolve_language("yor")
    assert iso == "yor"
    assert name == "Yoruba"

    iso, name = resolve_language("Swahili")
    assert iso == "swa"
    assert name == "Swahili"

    iso, name = resolve_language("HAUSA")
    assert iso == "hau"
    assert name == "Hausa"


def test_taxonomies_structure():
    # Verify core multi-task taxonomies
    assert "agriculture" in DOMAINS
    assert "healthcare" in DOMAINS
    assert "customer_support" in DOMAINS

    assert "positive" in SENTIMENTS
    assert "negative" in SENTIMENTS

    assert "joy" in EMOTIONS
    assert "frustration" in EMOTIONS

    assert "complaint" in INTENTS
    assert "cancellation_refund" in INTENTS
    assert "inquiry" in INTENTS

    assert len(LANGUAGES) >= 100
    assert "Twi" in LANGUAGES or "Akan" in LANGUAGES
    assert "Yoruba" in LANGUAGES


def test_languages_catalog():
    assert len(ISO_TO_NAME) >= 250
    assert len(NAME_TO_ISO) >= 250
    assert "zul" in ISO_TO_NAME
    assert ISO_TO_NAME["zul"] == "Zulu"
