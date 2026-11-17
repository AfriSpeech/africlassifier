#!/usr/bin/env python3
"""Push africlassifier artifacts to Hugging Face Hub under AfriSpeech/africlassifier.

Uploads:
1. Full 16-bit standalone merged model (model.safetensors, modules, tokenizer)
2. Quantized GGUF weights (Q8_0 for edge & mobile deployment)
3. Precomputed multi-task taxonomy index (taxonomies_index.pt)
4. LoRA adapter weights (adapter_model.safetensors, adapter_config.json)
5. Comprehensive evaluation benchmark results (benchmark_results.json)
6. Rich Hugging Face Model Card (README.md)
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


MODEL_CARD = """---
language:
- ak
- ee
- yo
- ha
- sw
- ig
- am
- om
- ln
- sn
- zu
- wo
- multilingual
license: apache-2.0
tags:
- audio-classification
- text-classification
- sentence-transformers
- sentence-similarity
- zero-shot-classification
- feature-extraction
- speech
- multimodal
- african-languages
- unsloth
pipeline_tag: feature-extraction
---

# africlassifier: CPU-First Multi-Task Foundation Classifier for African Speech & Text

**africlassifier** is an omnimodal zero-shot foundation model that classifies speech audio **or** text in **any African language** across 6 core NLP dimensions:
1. **Domain / Topic** (*Agriculture, Healthcare, Governance/Politics, Business/Finance, Education, Religion, Culture/Sports, Climate, Legal/Justice, Technology, Customer Support*)
2. **Sentiment** (*Positive, Negative, Neutral*)
3. **Emotion** (*Joy, Sadness, Anger, Fear, Surprise, Hope, Frustration, Neutral*)
4. **Intent Detection** (*Greeting, Inquiry, Complaint, Billing/Payment, Tech Support, Booking, Refund/Cancellation, Farewell*)
5. **Safety & Content Moderation** (*Hate Speech & Violence Detection vs Safe/Peaceful*)
6. **Language Identification** (*100+ African languages + English + French*)

Trained using High-Rank LoRA ($r=64, \\alpha=128$) on the Hopper H200 GPU over **796,481 sentence-level multimodal pairs**, aligning African speech audio directly with full, natural English sentences (CLIP representation principle).

---

## Model Release Artifacts in this Repository

| Artifact | File / Path | Size | Description |
|---|---|---|---|
| **Full Merged 16-bit** | `model.safetensors` | ~1.4 GB | Standalone merged bfloat16 weights. Works out of the box with `sentence-transformers` and `transformers`. |
| **Quantized GGUF (Q8_0)** | `gguf/africlassifier-q8_0.gguf` | ~290 MB | 8-bit quantized weights for edge, mobile, and llama.cpp deployment. |
| **Multimodal Projector** | `gguf/africlassifier-mmproj-bf16.gguf` | ~937 MB | Multimodal audio projector for GGUF runtime. |
| **Multi-Task Index** | `taxonomies_index.pt` | ~436 KB | Precomputed sentence-anchor embeddings for sub-15ms CPU multi-task scoring. |
| **LoRA Adapters** | `lora_adapters/` | ~99 MB | Lightweight LoRA adapter checkpoint for base `unsloth/embeddinggemma-2`. |
| **Benchmark Suite** | `benchmark_results.json` | ~45 KB | Detailed 16-language out-of-domain evaluation results. |

---

## Benchmark Results (16 African Languages)

Evaluated on **80 non-religious African speech clips and native texts across 16 African languages** sampled from **Google WaxalNLP** and **Google FLEURS**, with gold labels independently annotated by **Google Gemini 2.5 Flash**:

| Modality / Task | Metric | Score | Key Takeaways |
|---|---|---|---|
| **Audio Language ID** | Top-1 Accuracy | **47.5%** | 16-way classification from raw audio without ASR |
| **Audio Language ID** | Top-3 Accuracy | **60.0%** | Correct language within Top-3 candidates |
| **Audio Domain / Topic** | Top-3 Accuracy | **56.2%** | Broad domain alignment across 11 classes |
| **Audio Sentiment** | Accuracy | **28.8%** | Zero-shot speech sentiment |
| **Text Sentiment** | Accuracy | **53.8%** | Native African text sentiment |
| **Text Domain / Topic** | Top-1 Accuracy | **21.2%** | 11 semantic domain classes |
| **Text Domain / Topic** | Top-3 Accuracy | **38.8%** | Native African text |
| **Text Emotion** | Accuracy | **26.2%** | 8 emotion classes |
| **Safety / Violence Moderation** | Threat / Hostility Recall | **100.0%** (8/8) | Correctly caught all violent/warfare/killing statements |
| **Safety / Content Moderation** | Balanced Overall | **68.8%** (11/16) | Hate speech & violence vs peaceful expressions |
| **Customer Support Intent** | Cancellation / Refund | **100.0%** (4/4) | Across Twi, Yoruba, Swahili, Hausa |
| **Customer Support Intent** | Farewell | **100.0%** (4/4) | Across Twi, Yoruba, Swahili, Hausa |
| **Customer Support Intent** | Inquiry | **75.0%** (3/4) | Information seeking |
| **Customer Support Intent** | Top-2 Accuracy | **67.9%** (19/28) | High-recall chatbot intent routing |

### Evaluated Languages
- **West Africa**: Akan (Twi/Fante), Southern Dagaare, Ewe, Dagbani, Ikposo, Yoruba, Hausa
- **East Africa**: Swahili, Amharic, Oromo, Wolaytta, Luganda, Acholi
- **Central Africa**: Lingala
- **Southern Africa**: Shona, Zulu

---

## Quickstart

### 1. Using Standard `sentence-transformers` (Text Embeddings)
```python
from sentence_transformers import SentenceTransformer

# Load standalone 16-bit model directly from Hugging Face
model = SentenceTransformer("AfriSpeech/africlassifier")

sentences = [
    "Mema wo akye, me pɛ sɛ me nya nkyerɛkyerɛmu fa adwuma no ho.",  # Twi
    "Mo fẹ fagilee adehun mi ki ẹ si da owo mi pada.",                 # Yoruba
    "Ninataka kusitisha huduma hii naomba kurudishiwa pesa.",         # Swahili
]

embeddings = model.encode(sentences, normalize_embeddings=True)
print("Embeddings shape:", embeddings.shape)
```

### 2. Using `africlassifier` Python Package (Speech Audio & Text Multi-Task Classification)
```bash
pip install git+https://github.com/AfriSpeech/africlassifier.git
```

```python
from africlassifier import AfriClassifier

# Load CPU-first classifier (sub-15ms classification latency)
classifier = AfriClassifier.load(
    model_path="AfriSpeech/africlassifier",
    taxonomy_index_path="taxonomies_index.pt",
    device="cpu",  # or "cuda"
)

# Classify raw speech audio:
result = classifier.predict("speech.flac", top_k=3)
print(result.to_json(indent=2))

# Classify African text:
result = classifier.predict("Mo service no yɛ bɔne dodo, me pɛ me sika", top_k=3)
print(result.intent.top_1)   # cancellation_refund
print(result.safety.top_1)   # non_hate_speech
```

---

## Community Feedback & Contributions Welcome

More feedback, suggestions, and contributions are warmly welcome!
- **Taxonomy Refinement**: If you have domain-specific taxonomy needs (e.g., public health, civic engagement, agricultural extension), feel free to propose new sentence anchors.
- **Language Coverage**: Help expand anchors for additional African languages and regional dialects.
- **GitHub**: [https://github.com/AfriSpeech/africlassifier](https://github.com/AfriSpeech/africlassifier)
"""


def prepare_and_upload(
    repo_id: str = "AfriSpeech/africlassifier",
    merged_dir: str = "outputs/africlassifier_merged_16bit",
    lora_dir: str = "outputs/embeddinggemma_lora",
    taxonomies_index: str = "data/taxonomies_index.pt",
    benchmark_json: str = "data/benchmark_results.json",
    export_staging_dir: str = "outputs/hf_release_staging",
    private: bool = False,
):
    staging = Path(export_staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Preparing Release Package for Hugging Face Hub")
    print(f"Target Repository: {repo_id} (Public: {not private})")
    print(f"Staging Directory: {staging}")
    print("=" * 65)

    # 1. Copy merged 16-bit standalone model files
    merged_path = Path(merged_dir)
    if not merged_path.exists() or not (merged_path / "model.safetensors").exists():
        raise FileNotFoundError(f"Merged model not found at {merged_path}! Run save_pretrained_merged first.")

    print("\n[1/5] Copying standalone 16-bit model files...")
    for item in merged_path.iterdir():
        dst = staging / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    print("  ✓ Merged 16-bit model copied.")

    # 2. Copy GGUF quantized models (if available in /tmp/test_gguf)
    print("\n[2/5] Packaging Quantized GGUF files...")
    gguf_dir = staging / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    src_gguf = Path("/tmp/test_gguf")
    for f in ["embeddinggemma-2.Q8_0.gguf", "embeddinggemma-2.BF16-mmproj.gguf"]:
        src_f = src_gguf / f
        if src_f.exists():
            dst_name = "africlassifier-q8_0.gguf" if "Q8_0" in f else "africlassifier-mmproj-bf16.gguf"
            shutil.copy2(src_f, gguf_dir / dst_name)
            print(f"  ✓ Copied {dst_name} ({os.path.getsize(src_f)/(1024*1024):.1f} MB)")

    # 3. Copy LoRA adapter files
    print("\n[3/5] Packaging LoRA adapter checkpoint...")
    lora_staging = staging / "lora_adapters"
    lora_staging.mkdir(parents=True, exist_ok=True)
    lora_path = Path(lora_dir)
    for f in ["adapter_model.safetensors", "adapter_config.json"]:
        src_f = lora_path / f
        if src_f.exists():
            shutil.copy2(src_f, lora_staging / f)
            print(f"  ✓ Copied {f}")

    # 4. Copy multi-task taxonomy index & benchmark results
    print("\n[4/5] Copying multi-task taxonomy index & benchmark results...")
    idx_p = Path(taxonomies_index)
    if idx_p.exists():
        shutil.copy2(idx_p, staging / "taxonomies_index.pt")
        print(f"  ✓ Copied taxonomies_index.pt ({os.path.getsize(idx_p)/(1024):.1f} KB)")

    bench_p = Path(benchmark_json)
    if bench_p.exists():
        shutil.copy2(bench_p, staging / "benchmark_results.json")
        print("  ✓ Copied benchmark_results.json")

    # 5. Write Model Card README.md
    print("\n[5/5] Writing Hugging Face Model Card (README.md)...")
    readme_path = staging / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD)
    print("  ✓ Model Card generated.")

    # Upload to Hugging Face
    print("\n" + "=" * 65)
    print(f"Uploading Full Release Package to https://huggingface.co/{repo_id}...")
    print("=" * 65)
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")

    t0 = time.time()
    upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        commit_message="Release africlassifier: Full 16-bit standalone weights, Q8_0 GGUF, LoRA adapters & taxonomy index",
    )
    print(f"\n🎉 Successfully uploaded to https://huggingface.co/{repo_id} in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload africlassifier to Hugging Face.")
    parser.add_argument("--repo_id", type=str, default="AfriSpeech/africlassifier")
    parser.add_argument("--merged_dir", type=str, default="outputs/africlassifier_merged_16bit")
    parser.add_argument("--lora_dir", type=str, default="outputs/embeddinggemma_lora")
    parser.add_argument("--taxonomies_index", type=str, default="data/taxonomies_index.pt")
    parser.add_argument("--benchmark_json", type=str, default="data/benchmark_results.json")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    prepare_and_upload(
        repo_id=args.repo_id,
        merged_dir=args.merged_dir,
        lora_dir=args.lora_dir,
        taxonomies_index=args.taxonomies_index,
        benchmark_json=args.benchmark_json,
        private=args.private,
    )
