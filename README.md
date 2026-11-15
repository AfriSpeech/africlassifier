# speech2labels: African Speech to Concepts, Entities & Action Labels

**speech2labels** is a multimodal neural retrieval foundation model that maps speech audio in **any African language** directly to **English noun phrases** (via spaCy `doc.noun_chunks`), **African named entities**, and **action verbs** in a joint 768-dimensional embedding space using **EmbeddingGemma 2** and **Unsloth LoRA**.

No intermediate ASR transcription or language identification is required at inference time.

---

## 1. Key Innovations

1. **Direct Speech-to-Label Grounding:**
   - Base Architecture: `unsloth/embeddinggemma-2` (300M parameters).
   - Audio Encoder: 12-layer Conformer (25 frames/sec) adapted to African phonology and tonal variations using High-Rank LoRA ($r=64, \alpha=128$, `finetune_audio_layers=True`).
   - Text Projection: Gemma 2 language model layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Training Objective: `MultipleNegativesRankingLoss` (InfoNCE contrastive learning) with large in-batch negatives on an NVIDIA H200 (140 GB VRAM).
   - 100% Full Language Conditioning: Every concept prompt is gated with the natural language name (`"Twi: <label>"`, `"Yoruba: <label>"`, `"English: <label>"`, `"French: <label>"`).

2. **Parallel Corpus Alignment:**
   - **African Speech Audio:** `AfriSpeech/african-speech-public_v1` (100+ African languages, ~1,540 hours, ~430k clips).
   - **Parallel Text:** `AfriSpeech/african-corpus-jw` aligned by `verse_key`.
   - **English Reference:** `english_cache.csv` (~552,433 reference paragraphs).
   - **Cross-Lingual Anchoring:** Includes balanced African English (~5,000 clips) and French (~2,500 clips from FLEURS) to preserve acoustic boundaries and support code-switching.
   - **All Samples Included:** Retains all audio samples including numbers, Bible citations, and songs.

3. **Strict Concept & Entity Extraction:**
   - **Noun Phrases:** Strictly extracted using spaCy `doc.noun_chunks` as cohesive semantic units.
   - **Action Verbs:** Extracted via `token.pos_ == "VERB"` excluding generic auxiliaries.
   - **Open-Vocabulary Scaling:** Supports inference over an expanded taxonomy of **~696,000 concepts** (AfricaNouns + African Named Entities from codeswitch lexicons).

4. **Out-of-Domain Non-Religious Validation:**
   - Training loss and validation recall are evaluated directly on the **100% non-religious FLEURS dataset** across 13 African languages (covering news, science, history, and geography from FLoRes-101).

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
    device="cuda",
)

# Predict English concepts with full language conditioning (pass ISO or name)
result = classifier.predict(
    audio_input="sample_clip.flac",
    language="twi",     # Or "Twi", "yor", "hau", "Amharic", etc.
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
  "language_conditioned": {
    "iso": "twi",
    "name": "Twi"
  },
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
python -m speech2topics.infer --audio sample.flac --language twi --threshold 0.35 --top-k 10
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
