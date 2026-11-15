"""Extract salient English noun phrases and action verbs from text."""

import re
import spacy

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "one", "many", "much", "people", "someone", "anyone",
    "thing", "things", "way", "ways", "something", "anything", "nothing", "everything"
}

GENERIC_VERBS = {
    "be", "have", "do", "get", "go", "make", "take", "say", "come", "know",
    "see", "think", "look", "want", "give", "use", "find", "tell", "ask",
    "seem", "feel", "try", "leave", "call", "put", "keep", "let", "begin",
    "show", "hear", "run", "bring", "hold", "write", "stand", "set", "mean",
    "need", "include", "continue", "appear", "allow", "lead", "add"
}

_NLP = None


def get_spacy_model():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_concept_labels(text: str) -> dict:
    """Extract English noun phrases strictly using spaCy noun_chunks, plus action verbs.
    
    Returns:
        dict: {
            "noun_phrases": [{"label": str, "type": "noun_phrase"}],
            "verbs": [{"label": str, "type": "verb"}],
            "all_labels": [{"label": str, "type": str}]
        }
    """
    if not text or not isinstance(text, str):
        return {"noun_phrases": [], "verbs": [], "all_labels": []}

    nlp = get_spacy_model()
    doc = nlp(text)
    
    noun_phrases = []
    verbs = []
    seen = set()

    # 1. Noun phrases strictly from doc.noun_chunks
    for chunk in doc.noun_chunks:
        tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON") and t.text.lower() not in STOPWORDS]
        if not tokens:
            continue
        # Ensure the chunk contains a valid nominal head (NOUN or PROPN)
        if not any(t.pos_ in ("NOUN", "PROPN") for t in tokens):
            continue
        cleaned = " ".join(t.text for t in tokens).strip().lower()
        cleaned = re.sub(r"\s*-\s*", "-", cleaned)
        cleaned = re.sub(r"[^\w\s'-]", "", cleaned).strip()
        if len(cleaned) < 3 or cleaned in STOPWORDS or cleaned.isdigit():
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            noun_phrases.append({"label": cleaned, "type": "noun_phrase"})

    # 2. Informative action verbs
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ not in ("aux", "auxpass"):
            lemma = token.lemma_.lower()
            lemma = re.sub(r"[^\w'-]", "", lemma).strip()
            if len(lemma) >= 3 and lemma not in GENERIC_VERBS and lemma not in STOPWORDS and not lemma.isdigit():
                if lemma not in seen:
                    seen.add(lemma)
                    verbs.append({"label": lemma, "type": "verb"})

    all_labels = noun_phrases + verbs
    return {
        "noun_phrases": noun_phrases,
        "verbs": verbs,
        "all_labels": all_labels,
    }
