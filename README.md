# africlassifier: CPU-First Foundation Model for African Speech & Text Classification

**africlassifier** is an omnimodal neural retrieval foundation model that classifies speech audio **or** text in **any African language** directly into **English or French AfricaNouns** in a shared 768-dimensional embedding space using **EmbeddingGemma 2** and **Unsloth LoRA**.

Engineered **CPU-first** for fast, lightweight local deployment, with optional non-quantized GPU acceleration.

---

## 1. Key Capabilities

1. **Omnimodal Input (Speech OR Text):**
   - **Speech Audio:** Accepts `.wav`, `.flac`, `.mp3` in 100+ African languages.
   - **Native African Text:** Accepts raw sentences in any African orthography.
   - **Zero Intermediate Transcription:** Maps directly to conceptual meaning without speech-to-text or machine translation.

2. **Dual Target Languages (English or French AfricaNouns):**
   - User decides which language of noun phrases to retrieve (`target_language="en"` or `"fr"`). Default: **English (`en`)**.
   - Built directly from [AfriSpeech/africa-nouns](https://huggingface.co/datasets/AfriSpeech/africa-nouns) (2.77M English + 3.10M French verified African common nouns).
   - **No Named Entities:** Proper nouns, personal names, and demonyms are excluded; only pure common nouns and noun phrases are returned.
   - **No Verbs:** Strict focus on semantic concepts, topics, and noun phrases.

3. **CPU-First Engineering:**
   - **Fast CPU Inference:** Defaults to `device="cpu"` using optimized thread execution and optional dynamic INT8 quantization.
   - **GPU Support:** Deploy on `device="cuda"` with non-quantized weights when hardware acceleration is available.
   - **Sub-15ms Retrieval:** Precomputed L2-normalized vector indices (`data/indices/nouns_en_embeddings.pt` and `nouns_fr_embeddings.pt`) enable instant Top-K dot-product ranking across 100,000+ noun phrases.

4. **100% Full-Sentence Pretraining (The CLIP Principle):**
   - Fine-tuned on **796,481 full natural sentence pairs** from AfriSpeech JW parallel audio and text.
   - Evaluated out-of-domain on the **100% non-religious FLEURS benchmark** across 13 African languages.
   - Checkpoints achieve an out-of-domain evaluation loss of **~0.345** and training loss below **0.026**.

---

## 2. Directory Structure

```
africlassifier/
├── pyproject.toml
├── README.md
├── scripts/
│   ├── build_africlassifier_indices.py       # Encodes AfricaNouns (EN & FR) with final model
│   ├── build_sentence_foundation_dataset.py  # Full-sentence CLIP foundation builder
│   ├── export_mobile_package.py              # Mobile MRL vector quantization & bundling
│   ├── train_embeddinggemma_lora.py          # High-Rank LoRA Hopper training script
│   └── evaluate.py                           # Out-of-Domain FLEURS Recall@K evaluator
├── src/
│   └── africlassifier/
│       ├── __init__.py
│       ├── classifier.py                     # CPU-First inference engine (audio + text)
│       ├── languages.py                      # 308-language bidirectional ISO mapping
│       ├── server.py                         # FastAPI REST microservice (CPU-first)
│       └── infer.py                          # CLI tool emitting structured JSON
└── data/
    ├── indices/                              # Precomputed AfricaNouns vector indices
    │   ├── nouns_en_embeddings.pt            # 100,000 English AfricaNouns (float16)
    │   ├── nouns_en_metadata.json
    │   ├── nouns_fr_embeddings.pt            # 100,000 French AfricaNouns (float16)
    │   └── nouns_fr_metadata.json
    ├── clips/                                # 16 kHz mono FLAC clips
    └── train_pairs_sentence.jsonl            # Pure full-sentence training pairs
```

---

## 3. Quickstart & Usage

### Installation
```bash
git clone https://github.com/AfriSpeech/africlassifier.git
cd africlassifier
pip install -e .
# Optional for REST server:
pip install -e ".[server]"
```

### Python API (CPU-First by Default)
```python
from africlassifier import AfriClassifier

# Loads on CPU by default (fast & lightweight)
classifier = AfriClassifier.load(
    model_path="outputs/embeddinggemma_lora",
    indices_dir="data/indices",
    device="cpu",             # "cpu" (default) or "cuda"
    quantize_cpu=False,       # Set True for dynamic INT8 quantization on CPU
)

# 1. Classify African Speech Audio -> English Nouns (Default):
result = classifier.predict(
    input_data="sample_speech.flac",
    language="twi",           # Or "yoruba", "swahili", "aha", etc.
    target_language="en",     # "en" (default) or "fr"
    threshold=0.15,
    top_k=5,
)
print(result.to_json(indent=2))

# 2. Classify African Speech Audio -> French Nouns:
result_fr = classifier.predict(
    input_data="sample_speech.flac",
    language="twi",
    target_language="fr",     # Returns French AfricaNouns!
    top_k=5,
)
print(result_fr.to_json(indent=2))

# 3. Classify African Native Text -> English Nouns:
result_text = classifier.predict(
    input_data="Sɛ yɛbrɛ yɛn ho ase na yebu afoforo a, ɛma asomdwoe ba abusua no mu.",
    language="twi",
    target_language="en",
    top_k=5,
)
print(result_text.to_json(indent=2))
```

### CLI Usage
```bash
# CPU classification of audio into English AfricaNouns (default):
python -m africlassifier.infer --audio sample.flac --language twi --target-lang en

# CPU classification of audio into French AfricaNouns:
python -m africlassifier.infer --audio sample.flac --language twi --target-lang fr

# GPU classification with non-quantized weights:
python -m africlassifier.infer --audio sample.flac --language twi --device cuda

# Classify African text:
python -m africlassifier.infer --text "Kuayɔ mu nneɛma a ɛbɛma nnɔbae adɔ" --language twi
```

### Structured JSON Output
```json
{
  "input_type": "audio",
  "input_source": "sample_speech.flac",
  "language": {
    "iso": "twi",
    "name": "Twi"
  },
  "target_language": "en",
  "device_used": "cpu",
  "threshold": 0.15,
  "matched_count": 5,
  "inference_time_ms": 12.8,
  "matched_labels": [
    {
      "rank": 1,
      "label": "cocoa farming",
      "type": "noun_phrase",
      "score": 0.6421
    },
    {
      "rank": 2,
      "label": "crop yield",
      "type": "noun_phrase",
      "score": 0.5894
    },
    {
      "rank": 3,
      "label": "agricultural development",
      "type": "noun_phrase",
      "score": 0.5412
    }
  ],
  "top_prediction": {
    "rank": 1,
    "label": "cocoa farming",
    "type": "noun_phrase",
    "score": 0.6421
  }
}
```

---

## 4. FastAPI REST Microservice

```bash
# Launch on CPU (default):
python -m africlassifier.server

# Or on GPU:
USE_GPU=1 python -m africlassifier.server
```

**Endpoints:**
- `POST /classify/audio` (Form: `file`, `language`, `target_language="en"|"fr"`, `top_k=5`)
- `POST /classify/text`  (JSON: `{"text": "...", "language": "twi", "target_language": "en"}`)
