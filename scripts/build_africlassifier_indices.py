#!/usr/bin/env python3
"""Build English and French noun phrase indices from AfriSpeech/africa-nouns.

Using our final fine-tuned africlassifier model (outputs/embeddinggemma_lora):
1. Loads English AfricaNouns (africa-nouns-en.parquet).
2. Loads French AfricaNouns (africa-nouns-fr.parquet).
3. Pre-encodes both with our final fine-tuned model.
4. Saves precomputed vector indices and metadata for sub-5ms inference.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from unsloth import FastSentenceTransformer


def load_africa_nouns(parquet_path: str, min_count: int = 50, min_spread: int = 2, limit: int = None) -> list:
    """Load high-quality common noun phrases from AfricaNouns parquet."""
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    filtered = df[
        (df["total_count"] >= min_count)
        & (df["phrase"].str.len() >= 3)
        & (df["country_spread"] >= min_spread)
    ]
    if limit and len(filtered) > limit:
        filtered = filtered.head(limit)

    nouns = []
    seen = set()
    for _, row in filtered.iterrows():
        phrase = str(row["phrase"]).strip().lower()
        if phrase and phrase not in seen and not phrase.isdigit():
            seen.add(phrase)
            nouns.append({
                "label": phrase,
                "type": "noun_phrase",
                "total_count": int(row.get("total_count", 0)),
                "country_spread": int(row.get("country_spread", 0)),
            })
    print(f"  ✓ Loaded {len(nouns):,} common noun phrases.")
    return nouns


def encode_and_save_index(
    model,
    nouns: list,
    out_embeddings_path: Path,
    out_meta_path: Path,
    batch_size: int = 256,
    device: str = "cuda",
):
    texts = [item["label"] for item in nouns]
    print(f"Encoding {len(texts):,} phrases in batches of {batch_size}...")
    t0 = time.time()
    with torch.no_grad():
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    # Save as float16 tensor
    embs_fp16 = embs.half().cpu()
    torch.save(embs_fp16, out_embeddings_path)

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(nouns, f, indent=2, ensure_ascii=False)

    size_mb = out_embeddings_path.stat().st_size / 1e6
    print(f"  ✓ Saved index: {out_embeddings_path} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Build English & French indices from AfriSpeech/africa-nouns.")
    parser.add_argument("--model_path", type=str, default="outputs/embeddinggemma_lora", help="Fine-tuned model directory")
    parser.add_argument("--en_parquet", type=str, default="/mnt/volume_d2wey28/projects/africa-nouns/data/africa-nouns-en.parquet")
    parser.add_argument("--fr_parquet", type=str, default="/mnt/volume_d2wey28/projects/africa-nouns/data/africa-nouns-fr.parquet")
    parser.add_argument("--out_dir", type=str, default="data/indices", help="Output directory for precomputed indices")
    parser.add_argument("--min_count", type=int, default=50, help="Minimum corpus frequency")
    parser.add_argument("--min_spread", type=int, default=2, help="Minimum countries spread across")
    parser.add_argument("--limit_per_lang", type=int, default=100000, help="Max nouns per language (default 100k)")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load nouns from AfricaNouns (English and French)
    print("=" * 65)
    print("STEP 1: Loading Common Noun Phrases from AfriSpeech/africa-nouns")
    print("=" * 65)
    nouns_en = load_africa_nouns(args.en_parquet, min_count=args.min_count, min_spread=args.min_spread, limit=args.limit_per_lang)
    nouns_fr = load_africa_nouns(args.fr_parquet, min_count=args.min_count, min_spread=args.min_spread, limit=args.limit_per_lang)

    # 2. Load our final fine-tuned model
    print("\n" + "=" * 65)
    print(f"STEP 2: Loading Final Fine-Tuned Model ({args.model_path})")
    print("=" * 65)
    model = FastSentenceTransformer.from_pretrained(
        model_name=args.model_path,
        max_seq_length=1024,
        config_kwargs={"vision_config": None},
        full_finetuning=False,
    )
    model.eval()

    # 3. Encode and save English index
    print("\n" + "=" * 65)
    print("STEP 3: Encoding English AfricaNouns Index (Default)")
    print("=" * 65)
    encode_and_save_index(
        model=model,
        nouns=nouns_en,
        out_embeddings_path=out_dir / "nouns_en_embeddings.pt",
        out_meta_path=out_dir / "nouns_en_metadata.json",
        batch_size=args.batch_size,
    )

    # 4. Encode and save French index
    print("\n" + "=" * 65)
    print("STEP 4: Encoding French AfricaNouns Index")
    print("=" * 65)
    encode_and_save_index(
        model=model,
        nouns=nouns_fr,
        out_embeddings_path=out_dir / "nouns_fr_embeddings.pt",
        out_meta_path=out_dir / "nouns_fr_metadata.json",
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 65)
    print("ALL INDICES SUCCESSFULLY BUILT WITH FINAL FINE-TUNED MODEL")
    print("=" * 65)


if __name__ == "__main__":
    main()
