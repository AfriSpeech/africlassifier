#!/usr/bin/env python3
"""Add balanced English and French speech mapped to English concept labels.

This grounds the shared semantic space and teaches EmbeddingGemma 2 to distinguish
between English, French, and African languages:
- English Speech -> "English: <concept>"
- French Speech  -> "French: <concept>"
- African Speech -> "<African Language>: <concept>"
"""

import argparse
import glob
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from speech2topics.extractor import extract_concept_labels


def write_flac(audio_bytes: bytes, dst_path: str, target_sr: int = 16000) -> bool:
    try:
        with io.BytesIO(audio_bytes) as buf:
            data, sr = sf.read(buf, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = soxr.resample(data, sr, target_sr)
        sf.write(dst_path, data, target_sr, format="FLAC", subtype="PCM_16")
        return True
    except Exception:
        return False


def add_english_samples(
    data_dir: Path,
    clips_dir: Path,
    max_clips: int = 5000,
    parquet_pattern: str = "/mnt/volume_d2wey28/data/ghana-speech/English_eng/*.parquet",
) -> list:
    """Process African English speech from local parquets."""
    print(f"\nProcessing up to {max_clips:,} African English speech samples...")
    parquet_files = sorted(glob.glob(parquet_pattern))
    if not parquet_files:
        print(f"Warning: No English parquets found matching {parquet_pattern}")
        return []

    added_samples = []
    clip_count = 0

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            duration = float(row.get("duration", 0.0))
            if duration < 2.0 or duration > 25.0:
                continue

            text = row.get("text", "")
            if not text or len(text) < 15:
                continue

            concepts = extract_concept_labels(text)
            all_labels = concepts["all_labels"]
            if not all_labels:
                continue

            dst_audio = clips_dir / f"english_eng_{clip_count:06d}.flac"
            audio_data = row.get("audio", {})
            audio_bytes = audio_data.get("bytes") if isinstance(audio_data, dict) else None

            if audio_bytes and not dst_audio.exists():
                write_flac(audio_bytes, str(dst_audio))

            added_samples.append({
                "clip_path": str(dst_audio),
                "duration": round(duration, 2),
                "subset": "english_eng",
                "language": "English",
                "iso": "eng",
                "verse_key": f"eng_{clip_count}",
                "english_text": text,
                "label_objects": all_labels,
                "is_heldout": False,
            })
            clip_count += 1
            if clip_count >= max_clips:
                break

        if clip_count >= max_clips:
            break

    print(f"  ✓ Added {len(added_samples):,} English audio clips.")
    return added_samples


def add_french_samples(
    data_dir: Path,
    clips_dir: Path,
    max_clips: int = 3000,
) -> list:
    """Process French speech (FLEURS fr_fr), translating transcripts to English concepts."""
    print(f"\nProcessing up to {max_clips:,} French speech samples (FLEURS fr_fr)...")
    try:
        ds = load_dataset("google/fleurs", "fr_fr", split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"Warning: Failed to load FLEURS fr_fr: {e}")
        return []

    # Load Marian translation model (French -> English)
    print("Loading French -> English translator (Helsinki-NLP/opus-mt-fr-en)...")
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

    added_samples = []
    clip_count = 0

    batch_rows = []
    for row in tqdm(ds, desc="French Audio", total=max_clips):
        batch_rows.append(row)
        if len(batch_rows) >= 32:
            # Batch translate to English
            texts = [r["transcription"] for r in batch_rows]
            inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_length=128)
            eng_translations = [tok.decode(o, skip_special_tokens=True) for o in out]

            for r, eng_text in zip(batch_rows, eng_translations):
                concepts = extract_concept_labels(eng_text)
                all_labels = concepts["all_labels"]
                if not all_labels:
                    continue

                dst_audio = clips_dir / f"french_fra_{clip_count:06d}.flac"
                audio_bytes = r["audio"]["bytes"]
                if not dst_audio.exists():
                    write_flac(audio_bytes, str(dst_audio))

                added_samples.append({
                    "clip_path": str(dst_audio),
                    "duration": 5.0,
                    "subset": "french_fra",
                    "language": "French",
                    "iso": "fra",
                    "verse_key": f"fra_{clip_count}",
                    "english_text": eng_text,
                    "label_objects": all_labels,
                    "is_heldout": False,
                })
                clip_count += 1
                if clip_count >= max_clips:
                    break

            batch_rows = []
            if clip_count >= max_clips:
                break

    print(f"  ✓ Added {len(added_samples):,} French audio clips.")
    return added_samples


def main():
    parser = argparse.ArgumentParser(description="Add English and French samples to speech2topics dataset.")
    parser.add_argument("--data_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data")
    parser.add_argument("--english_clips", type=int, default=5000, help="Number of English clips to add")
    parser.add_argument("--french_clips", type=int, default=2500, help="Number of French clips to add")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    clips_dir = data_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    eng_samples = add_english_samples(data_dir, clips_dir, max_clips=args.english_clips)
    fra_samples = add_french_samples(data_dir, clips_dir, max_clips=args.french_clips)

    new_samples = eng_samples + fra_samples
    if not new_samples:
        print("No new samples to append.")
        return

    # Convert to training pairs
    train_split_ratio = 0.92
    np.random.seed(42)
    np.random.shuffle(new_samples)

    split_idx = int(len(new_samples) * train_split_ratio)
    new_train = new_samples[:split_idx]
    new_val = new_samples[split_idx:]

    print(f"\nAppending {len(new_train):,} training clips and {len(new_val):,} validation clips...")

    # Load existing catalog
    catalog_path = data_dir / "trained_labels.json"
    catalog_dict = {}
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                catalog_dict[item["label"]] = item

    new_train_pairs = []
    for s in new_train:
        for item in s["label_objects"]:
            lbl = item["label"]
            l_type = item["type"]
            new_train_pairs.append({
                "anchor": s["clip_path"],
                "positive": lbl,
                "label_type": l_type,
                "language": s["iso"],
                "verse_key": s["verse_key"],
            })
            if lbl not in catalog_dict:
                catalog_dict[lbl] = {"label": lbl, "type": l_type, "count": 0}
            catalog_dict[lbl]["count"] += 1

    new_val_pairs = []
    for s in new_val:
        for item in s["label_objects"]:
            new_val_pairs.append({
                "anchor": s["clip_path"],
                "positive": item["label"],
                "label_type": item["type"],
                "language": s["iso"],
            })

    # Append to existing files
    with open(data_dir / "train_pairs.jsonl", "a", encoding="utf-8") as f:
        for p in new_train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(data_dir / "val_pairs.jsonl", "a", encoding="utf-8") as f:
        for p in new_val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Update catalog
    catalog_list = sorted(catalog_dict.values(), key=lambda x: -x["count"])
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog_list, f, indent=2, ensure_ascii=False)

    print(f"Successfully added {len(new_train_pairs):,} English/French training pairs.")
    print(f"Updated catalog has {len(catalog_list):,} unique concepts.")


if __name__ == "__main__":
    main()
