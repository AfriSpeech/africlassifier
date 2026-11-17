#!/usr/bin/env python3
"""Export mobile-ready afri2labels artifacts: Merged weights, MRL-quantized vector index, and mobile metadata.

Modeled after the Bisa mobile deployment pipeline:
1. Merges LoRA weights into a standalone 16-bit / 8-bit model.
2. Embeds the expanded noun and concept catalog using the fine-tuned model.
3. Applies Matryoshka Representation Learning (MRL) truncation (e.g. 768-d -> 256-d).
4. Exports normalized float16 / int8 binary vectors for on-device vector search.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from unsloth import FastSentenceTransformer

from afri2labels.languages import resolve_language

SUPPORTED_DIMENSIONS = (128, 256, 512, 768)


def normalize_vectors(vectors: np.ndarray, dimension: Optional[int] = None) -> np.ndarray:
    """Optionally truncate via MRL and L2-normalize vectors, returning float32."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if dimension is not None and dimension < array.shape[-1]:
        array = array[:, :dimension]
    array = np.ascontiguousarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    np.divide(array, np.maximum(norms, 1e-12), out=array)
    return array


def export_mobile_package(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 65)
    logging.info("Exporting Mobile Deployment Package for afri2labels")
    logging.info("Model:     %s", args.model_path)
    logging.info("Catalog:   %s", args.catalog_path)
    logging.info("Dimension: %d (MRL Truncated)", args.dim)
    logging.info("=" * 65)

    # 1. Load model and merge LoRA weights
    logging.info("Loading fine-tuned model...")
    model = FastSentenceTransformer.from_pretrained(
        model_name=args.model_path,
        max_seq_length=1024,
        config_kwargs={"vision_config": None},
        full_finetuning=False,
    )
    model.eval()

    merged_model_dir = out_dir / "merged_model_16bit"
    if args.merge_weights and not merged_model_dir.exists():
        logging.info("Merging LoRA adapters into standalone 16-bit model at %s...", merged_model_dir)
        model.save_pretrained_merged(
            str(merged_model_dir),
            tokenizer=model.tokenizer,
            save_method="merged_16bit",
        )
        logging.info("Standalone merged model saved.")

    # 2. Load candidate concept list
    logging.info("Loading candidate concept catalog from %s...", args.catalog_path)
    with open(args.catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if args.limit and len(catalog) > args.limit:
        logging.info("Limiting catalog to top %d concepts for mobile bundle...", args.limit)
        catalog = catalog[:args.limit]

    labels = [item["label"] for item in catalog]
    logging.info("Total labels to embed: %d", len(labels))

    # Optional language prefix
    prefix_str = ""
    if args.language:
        _, lang_name = resolve_language(args.language)
        prefix_str = f"{lang_name or args.language.capitalize()}: "
        logging.info("Conditioning labels with language prefix: '%s'", prefix_str)

    texts_to_embed = [f"{prefix_str}{lbl}" for lbl in labels]

    # 3. Pre-embed using the fine-tuned model
    logging.info("Encoding candidate concepts into 768-d space (batch_size=%d)...", args.batch_size)
    t0 = time.time()
    with torch.no_grad():
        raw_embs = model.encode(
            texts_to_embed,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        ).float().cpu().numpy()
    logging.info("Encoded in %.1fs.", time.time() - t0)

    # 4. Truncate via MRL and normalize
    logging.info("Truncating to %d dimensions and L2-normalizing...", args.dim)
    norm_embs = normalize_vectors(raw_embs, dimension=args.dim)

    # 5. Save mobile artifacts
    vectors_file = out_dir / f"vectors_{args.dim}d.npy"
    vectors_bin = out_dir / f"vectors_{args.dim}d.bin"
    metadata_file = out_dir / "metadata.json"
    manifest_file = out_dir / "mobile_manifest.json"

    # Save numpy format
    np.save(vectors_file, norm_embs)
    # Save compact raw binary float16 format for direct C++/Kotlin memory-mapping
    norm_embs.astype(np.float16).tofile(vectors_bin)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    bin_size_mb = vectors_bin.stat().st_size / 1e6
    logging.info("Wrote binary vector index: %s (%.1f MB)", vectors_bin, bin_size_mb)

    mobile_manifest = {
        "model_name": "afri2labels",
        "base_model": "unsloth/embeddinggemma-2",
        "checkpoint_used": str(Path(args.model_path).resolve()),
        "native_dimension": 768,
        "exported_dimension": args.dim,
        "mrl_truncated": args.dim < 768,
        "total_labels": len(catalog),
        "language_conditioned": args.language,
        "vector_file_bin": vectors_bin.name,
        "vector_size_mb": round(bin_size_mb, 2),
        "metadata_file": metadata_file.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mobile_inference_runtimes": [
            "MNN (Alibaba Mobile Neural Network)",
            "Google LiteRT (.litertlm / .tflite)",
            "Sherpa-ONNX / ONNXRuntime Mobile"
        ]
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(mobile_manifest, f, indent=2)

    logging.info("=" * 65)
    logging.info("MOBILE EXPORT COMPLETE")
    logging.info("Export Dir:          %s", out_dir)
    logging.info("Total Concepts:      %d", len(catalog))
    logging.info("Vector Index Size:   %.1f MB (%d-dim float16)", bin_size_mb, args.dim)
    logging.info("Mobile Manifest:     %s", manifest_file)
    logging.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export mobile deployment package for afri2labels.")
    parser.add_argument("--model_path", type=str, default="outputs/checkpoint-8500", help="Model checkpoint to use")
    parser.add_argument("--catalog_path", type=str, default="data/codeswitch_nouns_catalog.json", help="africa-tts codeswitching noun phrases catalog")
    parser.add_argument("--output_dir", type=str, default="data/mobile_export", help="Output directory for mobile bundle")
    parser.add_argument("--dim", type=int, default=256, choices=SUPPORTED_DIMENSIONS, help="MRL dimension (128, 256, 512, 768)")
    parser.add_argument("--limit", type=int, default=100000, help="Max concepts to bundle (default 100k)")
    parser.add_argument("--language", type=str, default=None, help="Optional language prefix conditioning")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--merge_weights", action="store_true", help="Merge LoRA into standalone 16-bit model")
    args = parser.parse_args()
    export_mobile_package(args)
