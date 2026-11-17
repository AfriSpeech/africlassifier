#!/usr/bin/env python3
"""Precompute multi-task taxonomy embeddings using the fine-tuned africlassifier model."""

import argparse
import json
import time
from pathlib import Path

import torch
from unsloth import FastSentenceTransformer

from africlassifier.taxonomies import DOMAINS, EMOTIONS, INTENTS, LANGUAGES, SENTIMENTS


def precompute_taxonomies(model_path: str, out_file: str, device: str = "cuda"):
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Precomputing Multi-Task Taxonomy Embeddings for africlassifier")
    print(f"Model:  {model_path}")
    print(f"Output: {out_path}")
    print("=" * 65)

    model = FastSentenceTransformer.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        config_kwargs={"vision_config": None},
        full_finetuning=False,
    )
    model.eval()

    taxonomies = {
        "domain": DOMAINS,
        "sentiment": SENTIMENTS,
        "emotion": EMOTIONS,
        "intent": INTENTS,
        "language": LANGUAGES,
    }

    index_data = {}
    t0 = time.time()

    for task_name, task_dict in taxonomies.items():
        keys = list(task_dict.keys())
        descriptions = list(task_dict.values())
        print(f"Encoding {task_name:12s} ({len(keys):3d} classes)...")
        with torch.no_grad():
            embs = model.encode(
                descriptions,
                batch_size=128,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        index_data[task_name] = {
            "keys": keys,
            "embeddings": embs.half().cpu(),  # float16 tensor
        }

    torch.save(index_data, out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"\nSaved taxonomy index to {out_path} ({size_kb:.1f} KB) in {time.time()-t0:.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute multi-task taxonomy embeddings.")
    parser.add_argument("--model_path", type=str, default="outputs/embeddinggemma_lora")
    parser.add_argument("--out_file", type=str, default="data/taxonomies_index.pt")
    args = parser.parse_args()
    precompute_taxonomies(args.model_path, args.out_file)
