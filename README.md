# Speech2Topics: African Speech to English Noun Phrases & Action Verbs

A multimodal neural retrieval system that maps audio in **any African language** directly to **English noun phrases** (via spaCy `noun_chunks`) and **action verbs** in a joint 768-dimensional embedding space using **EmbeddingGemma 2** and **Unsloth LoRA**.

No intermediate ASR or machine translation is required at inference time.

---

## 1. Key Innovations

1. **Multimodal Concept Grounding:**
   - Base Model: `unsloth/embeddinggemma-2` (300M parameters).
   - Audio Encoder: 12-layer Conformer (25 frames/sec) adapted to African phonology using LoRA (`finetune_audio_layers=True`).
   - Text Projection: Gemma 2 language model layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Training Objective: `MultipleNegativesRankingLoss` with large in-batch negatives on NVIDIA H200 (140 GB VRAM).

2. **Parallel JW Corpus Alignment:**
   - **Audio Source:** `AfriSpeech/african-speech-public_v1` (142 African languages, ~2,268 hours).
   - **Parallel Text:** `AfriSpeech/african-corpus-jw` (Watchtower literature aligned across 100+ African languages by `verse_key`).
   - **English Reference:** `english_cache.csv` (~552,433 reference paragraphs).
   - **All Samples Included:** Includes full audio samples without dropping number/digit clips.

3. **Strict Concept Extraction:**
   - **Noun Phrases:** Strictly extracted using spaCy `doc.noun_chunks` (e.g. *"lack of confidence"*, *"shallow ocean water"*, *"god's promise"*).
   - **Action Verbs:** Informative predicate lemmas extracted via `token.pos_ == "VERB"` (e.g. *"suffer"*, *"trust"*, *"protect"*, *"worship"*).

---

## 2. Directory Structure

```
speech2topics/
├── pyproject.toml
├── README.md
├── scripts/
│   ├── build_jw_audio_noun_pairs.py   # Full parallel dataset builder
│   ├── train_embeddinggemma_lora.py    # Unsloth LoRA training script
│   └── evaluate.py                     # Recall@K (1, 5, 10) & MRR evaluator
├── src/
│   └── speech2topics/
│       ├── __init__.py
│       ├── classifier.py              # Inference engine (cosine similarity)
│       ├── extractor.py               # spaCy noun_chunks & verb extractor
│       └── infer.py                   # CLI tool emitting structured JSON
└── data/
    ├── clips/                         # 16 kHz mono FLAC clips
    ├── train_pairs.jsonl              # (audio, concept_label) training pairs
    ├── val_pairs.jsonl                # Validation pairs
    ├── test_clips.jsonl               # In-domain test clips
    ├── zero_shot_test_clips.jsonl     # Held-out African language test clips
    └── trained_labels.json            # Global catalog of all trained concepts
```

---

## 3. Inference Library Usage

### Python API
```python
from speech2topics import SpeechTopicClassifier

# Load fine-tuned model and trained concept catalog
classifier = SpeechTopicClassifier.load(
    model_path="outputs/embeddinggemma_lora",
    catalog_path="data/trained_labels.json",
    embeddings_cache_path="data/label_embeddings.pt",  # Optional precomputed cache
    device="cuda",
)

# Predict English concepts for any African speech audio
result = classifier.predict(
    audio_input="sample_clip.flac",
    threshold=0.35,     # Minimum cosine similarity
    top_k=10,           # Max returned labels
    filter_type=None,   # Optional: "noun_phrase" or "verb"
)

# Output structured JSON
print(result.to_json(indent=2))
```

### JSON Output Schema
```json
{
  "audio": "sample_clip.flac",
  "duration_seconds": 6.42,
  "threshold": 0.35,
  "matched_count": 4,
  "matched_labels": [
    {
      "label": "lack of confidence",
      "type": "noun_phrase",
      "score": 0.6421
    },
    {
      "label": "politicians",
      "type": "noun_phrase",
      "score": 0.5894
    },
    {
      "label": "trust",
      "type": "verb",
      "score": 0.5312
    },
    {
      "label": "suffer",
      "type": "verb",
      "score": 0.4851
    }
  ],
  "top_prediction": {
    "label": "lack of confidence",
    "type": "noun_phrase",
    "score": 0.6421
  }
}
```

### CLI Tool
```bash
python -m speech2topics.infer --audio sample.flac --threshold 0.35 --top-k 10
```

---

## 4. Detached Execution on H200

To ensure long-running data preparation and training jobs survive terminal disconnection:

### Monitor Dataset Build
```bash
ssh h200 'tmux attach -t build_speech2topics'
# Or inspect log:
ssh h200 'tail -f /mnt/volume_d2wey28/projects/speech2topics/data/build.log'
```

### Launch Training (Detached)
```bash
ssh h200 "tmux new-session -d -s train_speech2topics 'cd /mnt/volume_d2wey28/projects/speech2topics && export HF_HOME=/mnt/volume_d2wey28/hf-cache && .venv/bin/python scripts/train_embeddinggemma_lora.py --batch_size 128 --epochs 3 2>&1 | tee train.log'"
```
