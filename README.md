# africlassifier: CPU-First Multi-Task Foundation Classifier for African Speech & Text

**africlassifier** is an omnimodal zero-shot classification foundation model that classifies speech audio **or** text in **any African language** across 5 core NLP dimensions:
1. **Domain / Topic** *(Agriculture, Healthcare, Governance, Finance, Education, Religion, Tech, Customer Support)*
2. **Sentiment** *(Positive, Negative, Neutral)*
3. **Emotion** *(Joy, Sadness, Anger, Fear, Surprise, Hope, Frustration, Neutral)*
4. **Intent** *(Greeting, Inquiry, Complaint, Billing/Payment, Tech Support, Booking, Refund/Cancellation, Farewell)*
5. **Language Identification** *(100+ African languages + English + French)*

Engineered **CPU-first** for fast on-device and local server deployment, with optional non-quantized GPU acceleration.

---

## 1. Key Capabilities

- **Omnimodal Input (Speech OR Text):** Accepts raw African speech audio (`.wav`, `.flac`, `.mp3`) or native text sentences.
- **Top-1 by Default with Top-K Support:** Returns concise top-1 classification per task by default, with configurable `top_k` for ranked alternatives.
- **Automatic Language Identification:** If no language is provided, the model automatically detects the spoken or written language from 100+ African languages.
- **Sub-15ms CPU Inference:** Multi-task taxonomy vectors are pre-embedded into a 433 KB index (`data/taxonomies_index.pt`), enabling near-instant dot-product scoring on standard CPUs.

---

## 2. Directory Structure

```
africlassifier/
├── pyproject.toml
├── README.md
├── scripts/
│   ├── precompute_taxonomies.py              # Encodes multi-task taxonomies into fast index
│   ├── build_sentence_foundation_dataset.py  # Full-sentence CLIP foundation builder
│   ├── export_mobile_package.py              # Mobile MRL vector quantization & bundling
│   ├── train_embeddinggemma_lora.py          # High-Rank LoRA Hopper training script
│   └── evaluate.py                           # Out-of-Domain FLEURS Recall@K evaluator
├── src/
│   └── africlassifier/
│       ├── __init__.py
│       ├── classifier.py                     # CPU-First multi-task inference engine
│       ├── taxonomies.py                     # Domain, Sentiment, Emotion, Intent & Language anchors
│       ├── languages.py                      # 308-language bidirectional ISO mapping
│       ├── server.py                         # FastAPI REST microservice (CPU-first)
│       └── infer.py                          # CLI tool emitting structured JSON
└── data/
    ├── taxonomies_index.pt                   # Precomputed 433 KB multi-task vector index
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
    taxonomy_index_path="data/taxonomies_index.pt",
    device="cpu",  # "cpu" (default) or "cuda"
)

# 1. Spoken Audio Classification (Ahanta speech audio clip):
result_audio = classifier.predict(
    input_data="sample_speech.flac",
    top_k=1,  # Concise top-1 per task
)
print(result_audio.to_json(indent=2))

# 2. Native African Text Classification (Twi customer dispute):
result_text = classifier.predict(
    input_data="Me de me sika kɔ maa mo nanso mo service no yɛ dɔm pii, me pɛ sɛ mo gyae na mo de me sika ma me.",
    top_k=3,  # Top-3 ranked options with scores
)
print(result_text.to_json(indent=2))
```

### CLI Tool
```bash
# Classify speech audio on CPU:
python -m africlassifier.infer --audio speech.flac --device cpu

# Classify African text with top-3 alternatives:
python -m africlassifier.infer --text "Kuayɔ mu nneɛma a ɛbɛma nnɔbae adɔ" --top-k 3
```

### Structured Multi-Task JSON Output
```json
{
  "input_type": "audio",
  "input_source": "speech.flac",
  "duration_seconds": 6.73,
  "device_used": "cpu",
  "inference_time_ms": 10.96,
  "detected_language": {
    "label": "Ahanta",
    "score": 0.4557,
    "iso": "aha"
  },
  "domain": {
    "top_1": {
      "label": "religion_spirituality",
      "score": 0.106
    }
  },
  "sentiment": {
    "top_1": {
      "label": "positive",
      "score": 0.0755
    }
  },
  "emotion": {
    "top_1": {
      "label": "hope",
      "score": 0.0512
    }
  },
  "intent": {
    "top_1": {
      "label": "cancellation_refund",
      "score": 0.0609
    }
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
- `POST /classify/audio` (Form: `file`, `language` (optional), `top_k=1`)
- `POST /classify/text`  (JSON: `{"text": "...", "language": "twi", "top_k": 1}`)
