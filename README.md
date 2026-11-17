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
│   ├── build_sentence_foundation_dataset.py  # Pure full-sentence CLIP builder
│   ├── build_codeswitch_catalog.py           # Bundles 633k africa-tts noun phrases
│   ├── build_deployment_index.py             # Pre-encodes labels into fast tensor
│   ├── export_mobile_package.py              # Mobile MRL vector truncation & bundling
│   ├── train_embeddinggemma_lora.py          # High-Rank LoRA training script
│   └── evaluate.py                           # Recall@K (1, 5, 10) & MRR evaluator
├── src/
│   └── afri2labels/
│       ├── __init__.py
│       ├── classifier.py                     # Omnimodal inference engine (audio + text)
│       ├── extractor.py                      # spaCy noun_chunks & verb extractor
│       ├── languages.py                      # 308-language bidirectional ISO mapping
│       ├── server.py                         # High-performance FastAPI REST server
│       └── infer.py                          # CLI tool emitting structured JSON
└── data/
    ├── clips/                                # 16 kHz mono FLAC clips
    ├── train_pairs_sentence.jsonl            # Pure full-sentence training pairs
    ├── ood_val_pairs_sentence.jsonl          # Out-of-domain FLEURS validation pairs
    ├── codeswitch_nouns_catalog.json         # 633,697 africa-tts codeswitching noun phrases
    └── deployment_index/                     # Precomputed vector index for <5ms serving
```

---

## 3. Deployment Workflow (Pre-Embedded Vector Search)

### Step 1: Pre-Encode the ~633k africa-tts Noun Phrases (Offline, One-Time)
```bash
python scripts/build_deployment_index.py \
    --model_path outputs/checkpoint-34500 \
    --catalog_path data/codeswitch_nouns_catalog.json \
    --out_dir data/deployment_index
```
This generates `data/deployment_index/label_embeddings.pt` (normalized float16 tensor, ~1 GB) once.

### Step 2: High-Performance Serving (< 5ms Query Latency)

#### A. Python API
```python
from afri2labels import Afri2LabelsClassifier

classifier = Afri2LabelsClassifier.load(
    model_path="outputs/checkpoint-34500",
    catalog_path="data/codeswitch_nouns_catalog.json",
    embeddings_cache_path="data/deployment_index/label_embeddings.pt",
    device="cuda",
)

# 1. Spoken Audio Query:
result = classifier.predict("sample.flac", language="twi", top_k=5)

# 2. African Native Text Query:
result = classifier.predict("Sɛ yɛbrɛ yɛn ho ase na yebu afoforo a...", language="twi", top_k=5)

print(result.to_json(indent=2))
```

#### B. FastAPI REST Microservice
```bash
python -m afri2labels.server
# Listening on http://0.0.0.0:8000
# Endpoints: POST /predict/audio, POST /predict/text
```

### JSON Output Schema
```json
{
  "input_type": "audio",
  "input_source": "sample_clip.flac",
  "duration_seconds": 6.42,
  "threshold": 0.35,
  "language_conditioned": {
    "iso": "twi",
    "name": "Twi"
  },
  "matched_count": 3,
  "matched_labels": [
    {
      "rank": 1,
      "label": "cassava harvest",
      "type": "noun_phrase",
      "score": 0.7421
    },
    {
      "rank": 2,
      "label": "farmers",
      "type": "noun_phrase",
      "score": 0.6914
    },
    {
      "rank": 3,
      "label": "crop yield",
      "type": "noun_phrase",
      "score": 0.6542
    }
  ],
  "top_prediction": {
    "rank": 1,
    "label": "cassava harvest",
    "type": "noun_phrase",
    "score": 0.7421
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
