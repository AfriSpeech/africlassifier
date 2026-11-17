"""Unit tests for afri2labels inference, languages, and concept extraction."""

import json
import pytest
from afri2labels.extractor import extract_concept_labels
from afri2labels.languages import ISO_TO_NAME, NAME_TO_ISO, resolve_language


def test_language_resolution():
    # Test ISO to Name
    iso, name = resolve_language("twi")
    assert iso == "twi"
    assert name == "Twi"

    iso, name = resolve_language("yor")
    assert iso == "yor"
    assert name == "Yoruba"

    # Test Name to ISO
    iso, name = resolve_language("Swahili")
    assert iso == "swa"
    assert name == "Swahili"

    iso, name = resolve_language("Amharic")
    assert iso == "amh"
    assert name == "Amharic"

    # Test Case Insensitivity
    iso, name = resolve_language("HAUSA")
    assert iso == "hau"
    assert name == "Hausa"


def test_concept_extraction():
    sample_text = (
        "Many cocoa farmers in the western region are adopting modern agricultural practices "
        "to increase crop yields and protect forest reserves."
    )
    result = extract_concept_labels(sample_text)

    # Check structure
    assert "noun_phrases" in result
    assert "verbs" in result
    assert "all_labels" in result

    # Check extracted noun phrases (strictly from doc.noun_chunks)
    np_labels = [item["label"] for item in result["noun_phrases"]]
    assert any("cocoa farmers" in lbl for lbl in np_labels)
    assert any("agricultural practices" in lbl for lbl in np_labels)
    assert any("crop yields" in lbl for lbl in np_labels)

    # Check extracted action verbs
    verb_labels = [item["label"] for item in result["verbs"]]
    assert "adopt" in verb_labels or "increase" in verb_labels or "protect" in verb_labels


def test_languages_catalog():
    assert len(ISO_TO_NAME) >= 250
    assert len(NAME_TO_ISO) >= 250
    assert "zul" in ISO_TO_NAME
    assert ISO_TO_NAME["zul"] == "Zulu"
