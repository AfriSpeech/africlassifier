#!/usr/bin/env python3
"""Build complete aligned (African Speech Audio -> English Noun Phrases & Verbs) dataset from JW sources.

Features:
1. Complete Dataset: uses ALL audio samples from AfriSpeech (including number/digit clips).
2. Direct Parquet Ingestion: pulls audio bytes directly from AfriSpeech/african-speech-public_v1.
3. Accurate Parallel Alignment: matches African transcripts to African-Corpus-JW to get verse_key.
4. English Concepts: maps verse_key -> English paragraph in english_cache.csv.
5. Strict spaCy Noun Chunks: extracts noun phrases strictly via doc.noun_chunks, plus action verbs.
6. Fast Multithreaded Slicing: saves 16 kHz mono FLAC clips in parallel using soxr.
7. Topic-Disjoint Splits: splits by verse_key into train, val, and held-out zero-shot test sets.
8. Output Catalog: generates trained_labels.json with all unique concepts.
"""

import argparse
import glob
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from speech2topics.extractor import extract_concept_labels

ISO3_TO_ISO1 = {
    "wol": "wo", "tsn": "tn", "kon": "kg", "sot": "st", "nya": "ny", "amh": "am",
    "orm": "om", "swa": "sw", "tir": "ti", "yor": "yo", "zul": "zu", "xho": "xh",
    "som": "so", "sna": "sn", "mlg": "mg", "lin": "ln", "kin": "rw", "run": "rn",
    "hau": "ha", "ibo": "ig", "ful": "ff", "aar": "aa", "tso": "ts", "twi": "tw",
    "aka": "ak", "bam": "bm", "ewe": "ee", "lug": "lg", "ssw": "ss", "ven": "ve",
    "nbl": "nr", "ndo": "ng", "her": "hz", "kua": "kj", "sag": "sg",
}


def norm_prefix(text: str, length: int = 40) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF]", "", text.lower())
    return cleaned[:length]


def write_flac_clip(audio_bytes: bytes, dst_path: str, target_sr: int = 16000) -> bool:
    """Decode audio bytes, resample to 16 kHz mono, and save as 16-bit FLAC."""
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


def build_dataset(args):
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    cache_dir = Path(args.cache_dir)
    parquet_cache_dir = Path(args.parquet_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STEP 1: Loading English Anchor Text & Corpus Catalog")
    print("=" * 65)
    eng_cache_file = cache_dir / "english_cache.csv"
    if not eng_cache_file.exists():
        print("Downloading english_cache.csv...")
        hf_hub_download(
            repo_id="AfriSpeech/african-corpus-jw",
            filename="english_cache.csv",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )

    print(f"Loading {eng_cache_file}...")
    df_eng = pd.read_csv(eng_cache_file).dropna(subset=["verse_key", "eng"])
    eng_dict: Dict[str, str] = dict(zip(df_eng["verse_key"], df_eng["eng"]))
    print(f"Loaded {len(eng_dict):,} English reference paragraphs.")

    # Find all CSV files in cache
    csv_files = [f for f in os.listdir(cache_dir) if f.endswith(".csv") and f != "english_cache.csv"]
    iso_to_csv = {}
    for f in csv_files:
        m = re.match(r"^(?:.+)_([a-zA-Z0-9]+)_v\d+\.csv$", f)
        if m:
            iso_to_csv[m.group(1).lower()] = f

    print("\n" + "=" * 65)
    print("STEP 2: Discovering AfriSpeech Audio Shards")
    print("=" * 65)
    api = HfApi()
    info = api.dataset_info("AfriSpeech/african-speech-public_v1", files_metadata=True)
    all_parquet_files = [f.rfilename for f in info.siblings if f.rfilename.endswith(".parquet")]

    # Group parquets by subset (language)
    subset_to_parquets: Dict[str, List[str]] = {}
    for pf in all_parquet_files:
        subset = pf.split("/")[0]
        subset_to_parquets.setdefault(subset, []).append(pf)

    print(f"Discovered {len(all_parquet_files)} parquet shards across {len(subset_to_parquets)} languages.")

    # Match subsets to text CSVs
    subsets_to_process = []
    for s in sorted(subset_to_parquets.keys()):
        parts = s.split("_")
        iso = parts[-1].lower()
        alt_iso = ISO3_TO_ISO1.get(iso, "")
        name_prefix = parts[0].lower()

        matched_csv = None
        if iso in iso_to_csv:
            matched_csv = iso_to_csv[iso]
        elif alt_iso in iso_to_csv:
            matched_csv = iso_to_csv[alt_iso]
        else:
            candidates = [f for f in csv_files if f.lower().startswith(name_prefix)]
            if candidates:
                matched_csv = candidates[0]

        if matched_csv:
            subsets_to_process.append((s, matched_csv, subset_to_parquets[s]))

    print(f"Matching languages with parallel text: {len(subsets_to_process)} / {len(subset_to_parquets)}")

    if args.languages:
        target_set = set(args.languages.split(","))
        subsets_to_process = [item for item in subsets_to_process if item[0] in target_set]
        print(f"Filtered to {len(subsets_to_process)} requested languages.")

    heldout_set = set(args.heldout_langs.split(",")) if args.heldout_langs else set()

    print("\n" + "=" * 65)
    print("STEP 3: Processing Audio Samples & Extracting Noun Chunks + Verbs")
    print("=" * 65)

    all_matched_samples = []
    total_audio_hours = 0.0

    # Thread pool for saving audio clips
    clip_executor = ThreadPoolExecutor(max_workers=args.workers)
    pending_writes = []

    # Cache pre-extracted concept labels for unique verse_keys to make processing instant
    vk_concept_cache: Dict[str, dict] = {}

    for subset, matched_csv, pq_list in tqdm(subsets_to_process, desc="Languages"):
        parts = subset.split("_")
        iso = parts[-1].lower()

        # Load local text mapping
        local_csv_path = cache_dir / matched_csv
        try:
            df_local = pd.read_csv(local_csv_path).dropna(subset=["verse_key", "local"])
            local_map = {}
            for vk, txt in zip(df_local["verse_key"], df_local["local"]):
                k = norm_prefix(txt)
                if k and len(k) >= 12:
                    local_map[k] = vk
        except Exception:
            continue

        lang_clips = 0
        lang_hours = 0.0

        for pq_rel_path in pq_list:
            # Download shard if not cached
            try:
                local_pq = hf_hub_download(
                    repo_id="AfriSpeech/african-speech-public_v1",
                    filename=pq_rel_path,
                    repo_type="dataset",
                    local_dir=str(parquet_cache_dir),
                )
                df_pq = pd.read_parquet(local_pq)
            except Exception as e:
                continue

            for _, row in df_pq.iterrows():
                duration = float(row.get("duration", 0.0))
                if duration < args.min_duration or duration > args.max_duration:
                    continue

                text = row.get("text", "")
                k = norm_prefix(text)
                vk = local_map.get(k)
                if not vk or vk not in eng_dict:
                    continue

                # Extract concept labels strictly using spaCy noun_chunks + action verbs
                if vk not in vk_concept_cache:
                    eng_text = eng_dict[vk]
                    vk_concept_cache[vk] = {
                        "text": eng_text,
                        "concepts": extract_concept_labels(eng_text),
                    }

                cached = vk_concept_cache[vk]
                eng_text = cached["text"]
                concepts = cached["concepts"]
                noun_phrases = concepts["noun_phrases"]
                verbs = concepts["verbs"]
                all_labels = concepts["all_labels"]

                if not all_labels:
                    continue

                # Audio clip path
                clip_filename = f"{subset}_{lang_clips:06d}.flac"
                dst_audio = clips_dir / clip_filename

                audio_data = row.get("audio", {})
                audio_bytes = audio_data.get("bytes") if isinstance(audio_data, dict) else None

                if audio_bytes and not dst_audio.exists():
                    f = clip_executor.submit(write_flac_clip, audio_bytes, str(dst_audio))
                    pending_writes.append(f)

                sample = {
                    "clip_path": str(dst_audio),
                    "duration": round(duration, 2),
                    "subset": subset,
                    "language": subset,
                    "iso": iso,
                    "verse_key": vk,
                    "english_text": eng_text,
                    "noun_phrases": [n["label"] for n in noun_phrases],
                    "verbs": [v["label"] for v in verbs],
                    "all_labels": [item["label"] for item in all_labels],
                    "label_objects": all_labels,
                    "is_heldout": (subset in heldout_set or iso in heldout_set),
                }
                all_matched_samples.append(sample)
                lang_clips += 1
                lang_hours += duration / 3600.0
                total_audio_hours += duration / 3600.0

                if args.max_clips_per_lang and lang_clips >= args.max_clips_per_lang:
                    break

            if args.max_clips_per_lang and lang_clips >= args.max_clips_per_lang:
                break

        print(f"  ✓ {subset:25s}: {lang_clips:5d} clips ({lang_hours:.1f} h)")

    # Wait for audio writes to complete
    print("\nFinalizing audio clip encoding...")
    for f in tqdm(pending_writes, desc="Writing FLACs", leave=False):
        f.result()
    clip_executor.shutdown()

    print("\n" + "=" * 65)
    print("STEP 4: Generating Train, Validation & Zero-Shot Test Splits")
    print("=" * 65)
    print(f"Total aligned audio clips: {len(all_matched_samples):,} ({total_audio_hours:.1f} hours)")

    train_pool = [s for s in all_matched_samples if not s["is_heldout"]]
    heldout_pool = [s for s in all_matched_samples if s["is_heldout"]]

    # Split by verse_key to prevent topic overlap between train and test
    all_vks = sorted(list({s["verse_key"] for s in train_pool}))
    np.random.seed(42)
    np.random.shuffle(all_vks)

    n_val = int(len(all_vks) * args.val_ratio)
    n_test = int(len(all_vks) * args.test_ratio)

    val_vks = set(all_vks[:n_val])
    test_vks = set(all_vks[n_val : n_val + n_test])
    train_vks = set(all_vks[n_val + n_test :])

    train_clips = [s for s in train_pool if s["verse_key"] in train_vks]
    val_clips = [s for s in train_pool if s["verse_key"] in val_vks]
    test_clips = [s for s in train_pool if s["verse_key"] in test_vks]

    print(f"Splits by verse_key:")
    print(f"  Train:     {len(train_clips):6d} clips ({len(train_vks):,} unique paragraphs)")
    print(f"  Val:       {len(val_clips):6d} clips ({len(val_vks):,} unique paragraphs)")
    print(f"  Test:      {len(test_clips):6d} clips ({len(test_vks):,} unique paragraphs)")
    if heldout_pool:
        print(f"  Held-out (Zero-Shot Langs): {len(heldout_pool):6d} clips")

    # Generate contrastive pairs (anchor: audio, positive: noun phrase or action verb)
    train_pairs = []
    unique_labels_dict = {}

    for s in train_clips:
        for item in s["label_objects"]:
            lbl = item["label"]
            l_type = item["type"]
            train_pairs.append({
                "anchor": s["clip_path"],
                "positive": lbl,
                "label_type": l_type,
                "language": s["iso"],
                "verse_key": s["verse_key"],
            })
            if lbl not in unique_labels_dict:
                unique_labels_dict[lbl] = {"label": lbl, "type": l_type, "count": 0}
            unique_labels_dict[lbl]["count"] += 1

    val_pairs = []
    for s in val_clips:
        for item in s["label_objects"]:
            val_pairs.append({
                "anchor": s["clip_path"],
                "positive": item["label"],
                "label_type": item["type"],
                "language": s["iso"],
            })

    # Save to disk
    with open(out_dir / "train_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(out_dir / "val_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(out_dir / "test_clips.jsonl", "w", encoding="utf-8") as f:
        for s in test_clips:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if heldout_pool:
        with open(out_dir / "zero_shot_test_clips.jsonl", "w", encoding="utf-8") as f:
            for s in heldout_pool:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Save label catalog
    catalog_list = sorted(unique_labels_dict.values(), key=lambda x: -x["count"])
    with open(out_dir / "trained_labels.json", "w", encoding="utf-8") as f:
        json.dump(catalog_list, f, indent=2, ensure_ascii=False)

    num_nps = sum(1 for item in catalog_list if item["type"] == "noun_phrase")
    num_verbs = sum(1 for item in catalog_list if item["type"] == "verb")

    print("\n" + "=" * 65)
    print("DATASET BUILD SUMMARY")
    print("=" * 65)
    print(f"Output directory:            {out_dir}")
    print(f"Total audio clips:           {len(all_matched_samples):,} ({total_audio_hours:.1f} hours)")
    print(f"Total training pairs:        {len(train_pairs):,}")
    print(f"Total validation pairs:      {len(val_pairs):,}")
    print(f"Unique concepts in catalog:  {len(catalog_list):,}")
    print(f"  - Noun phrases (spaCy chunks): {num_nps:,}")
    print(f"  - Action verbs:                {num_verbs:,}")
    print(f"Label catalog saved to:      {out_dir / 'trained_labels.json'}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build complete aligned audio-to-concepts dataset from JW sources.")
    parser.add_argument("--out_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data")
    parser.add_argument("--cache_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data/jw_corpus_cache")
    parser.add_argument("--parquet_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data/parquet_cache")
    parser.add_argument("--max_clips_per_lang", type=int, default=None, help="Max clips per language (default None = all samples)")
    parser.add_argument("--min_duration", type=float, default=1.5, help="Min duration in seconds")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max duration in seconds")
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated subset names (default all matching languages)")
    parser.add_argument("--heldout_langs", type=str, default="zulu_zul,somali_som", help="Held-out zero-shot test languages")
    parser.add_argument("--val_ratio", type=float, default=0.08)
    parser.add_argument("--test_ratio", type=float, default=0.08)
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers for audio encoding")

    args = parser.parse_args()
    build_dataset(args)
