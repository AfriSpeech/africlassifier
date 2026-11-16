#!/usr/bin/env python3
"""Build Pre-Embedded Vector Index for afri2labels Deployment.

Encodes the expanded noun and concept list once using the fine-tuned afri2labels model
into a normalized float16 tensor, enabling sub-5ms Top-K search at inference time.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from unsloth import FastSentenceTransformer

from afri2labels.languages import resolve_language


def build_deployment_index(
    model_path: str,
    catalog_path: str,
    out_dir: str,
    language_prefix: Optional[str] = None,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Building Deployment Vector Index for afri2labels")
    print(f"Model:    {model_path}")
    print(f"Catalog:  {catalog_path}")
    print(f"Device:   {device}")
    print("=" * 65)

    # 1. Load catalog
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"Loaded {len(catalog):,} candidate concepts from {catalog_path}")

    # Resolve optional language prefix
    prefix_str = ""
    if language_prefix:
        _, lang_name = resolve_language(language_prefix)
        prefix_str = f"{lang_name or language_prefix.capitalize()}: "
        print(f"Applying Language Conditioning Prefix: '{prefix_str}'")

    texts_to_encode = [f"{prefix_str}{item['label']}" for item in catalog]

    # 2. Load model
    print("\nLoading model for index encoding...")
    t0 = time.time()
    model = FastSentenceTransformer.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        config_kwargs={"vision_config": None},
        full_finetuning=False,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s.")

    # 3. Precompute normalized embeddings
    print(f"\nEncoding {len(texts_to_encode):,} concepts in batches of {batch_size}...")
    t_start = time.time()
    with torch.no_grad():
        embeddings = model.encode(
            texts_to_encode,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    # Save as float16 tensor for fast dot-product and compact memory (~75 MB per 50k labels)
    embeddings_fp16 = embeddings.half().cpu()
    index_file = out_dir / "label_embeddings.pt"
    meta_file = out_dir / "label_metadata.json"

    print(f"\nSaving vector index to {index_file} ({embeddings_fp16.element_size() * embeddings_fp16.nelement() / 1e6:.1f} MB)...")
    torch.save(embeddings_fp16, index_file)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print("DEPLOYMENT INDEX BUILD COMPLETE")
    print(f"Total Labels Indexed: {len(catalog):,}")
    print(f"Vector Tensor:        {index_file}")
    print(f"Metadata File:        {meta_file}")
    print(f"Total Encoding Time:  {time.time()-t_start:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build pre-embedded vector index for afri2labels.")
    parser.add_argument("--model_path", type=str, default="outputs/checkpoint-1000", help="Path to model checkpoint")
    parser.add_argument("--catalog_path", type=str, default="data/expanded_labels.json", help="Path to concepts JSON")
    parser.add_argument("--out_dir", type=str, default="data/deployment_index", help="Output directory for index")
    parser.add_argument("--language", type=str, default=None, help="Optional language prefix (e.g. twi, yor)")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    build_deployment_index(
        model_path=args.model_path,
        catalog_path=args.catalog_path,
        out_dir=args.out_dir,
        language_prefix=args.language,
        batch_size=args.batch_size,
    )
