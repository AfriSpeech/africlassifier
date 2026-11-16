#!/usr/bin/env python3
"""Build Pure Full-Sentence Semantic Foundation Dataset for afri2labels (CLIP / CLAP Principle).

100% Full-Sentence Natural Language Semantics:
- Audio(African Speech) ──► "{Language}: {Full English Translation}"
- Text("{Language}: {African Native Text}") ──► "{Language}: {Full English Translation}"

Establishes pure, uncompromised global semantic geometry across spoken audio and text.
Enables open-ended zero-shot retrieval for any label, topic, noun, verb, or complex query.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

from speech2topics.languages import resolve_language

ISO3_TO_ISO1 = {
    "wol": "wo", "tsn": "tn", "kon": "kg", "sot": "st", "nya": "ny", "amh": "am",
    "orm": "om", "swa": "sw", "tir": "ti", "yor": "yo", "zul": "zu", "xho": "xh",
    "som": "so", "sna": "sn", "mlg": "mg", "lin": "ln", "kin": "rw", "run": "rn",
    "hau": "ha", "ibo": "ig", "ful": "ff", "aar": "aa", "tso": "ts", "twi": "tw",
    "aka": "ak", "bam": "bm", "ewe": "ee", "lug": "lg", "ssw": "ss", "ven": "ve",
    "nbl": "nr", "ndo": "ng", "her": "hz", "kua": "kj", "sag": "sg",
}


def build_sentence_dataset(args):
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    print("=" * 65)
    print("STEP 1: Indexing Full English Sentences & African Parallel Text")
    print("=" * 65)

    eng_cache_file = cache_dir / "english_cache.csv"
    df_eng = pd.read_csv(eng_cache_file).dropna(subset=["verse_key", "eng"])
    eng_dict: Dict[str, str] = dict(zip(df_eng["verse_key"].astype(str), df_eng["eng"]))
    print(f"Loaded {len(eng_dict):,} English reference sentences.")

    csv_files = [f for f in os.listdir(cache_dir) if f.endswith(".csv") and f != "english_cache.csv"]
    iso_vk_to_text: Dict[Tuple[str, str], str] = {}

    for f in tqdm(csv_files, desc="Indexing African CSVs"):
        m = re.match(r"^(?:.+)_([a-zA-Z0-9]+)_v\d+\.csv$", f)
        if not m:
            continue
        code = m.group(1).lower()
        try:
            df = pd.read_csv(cache_dir / f).dropna(subset=["verse_key", "local"])
            for vk, txt in zip(df["verse_key"], df["local"]):
                clean_t = str(txt).strip()
                if len(clean_t) >= 5:
                    iso_vk_to_text[(code, str(vk))] = clean_t
        except Exception:
            continue

    print(f"Indexed {len(iso_vk_to_text):,} (code, verse_key) -> African text entries.")

    print("\n" + "=" * 65)
    print("STEP 2: Generating Pure Full-Sentence Training Pairs (CLIP Principle)")
    print("=" * 65)

    train_file = data_dir / "train_pairs.jsonl"
    out_train_sentence = data_dir / "train_pairs_sentence.jsonl"

    clip_records: Dict[str, dict] = {}

    with open(train_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading Unique Clips"):
            r = json.loads(line)
            audio_path = r["anchor"]
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                continue

            iso = (r.get("language") or r.get("subset", "").split("_")[-1]).strip().lower()
            vk = str(r.get("verse_key", ""))

            if audio_path not in clip_records:
                clip_records[audio_path] = {
                    "audio_path": audio_path,
                    "iso": iso,
                    "verse_key": vk,
                }

    print(f"Found {len(clip_records):,} unique training audio clips.")

    sentence_train_pairs = []
    seen_text_sentences: Set[Tuple[str, str]] = set()

    for audio_path, info in tqdm(clip_records.items(), desc="Generating Sentence Pairs"):
        iso = info["iso"]
        vk = info["verse_key"]
        _, lang_name = resolve_language(iso)
        prefix = lang_name or "African"

        eng_sentence = eng_dict.get(vk)
        alt_code = ISO3_TO_ISO1.get(iso, iso)
        lookup_key = (iso, vk) if (iso, vk) in iso_vk_to_text else (alt_code, vk)
        afri_sentence = iso_vk_to_text.get(lookup_key)

        if not eng_sentence:
            continue

        # 1. Speech Audio -> Full English Sentence
        sentence_train_pairs.append({
            "anchor": audio_path,
            "positive": f"{prefix}: {eng_sentence}",
            "modality": "audio",
        })

        # 2. African Native Text -> Full English Sentence
        if afri_sentence and (prefix, afri_sentence[:60]) not in seen_text_sentences:
            seen_text_sentences.add((prefix, afri_sentence[:60]))
            sentence_train_pairs.append({
                "anchor": f"{prefix}: {afri_sentence}",
                "positive": f"{prefix}: {eng_sentence}",
                "modality": "text",
            })

    random.seed(42)
    random.shuffle(sentence_train_pairs)

    num_audio = sum(1 for p in sentence_train_pairs if p["modality"] == "audio")
    num_text = sum(1 for p in sentence_train_pairs if p["modality"] == "text")

    print(f"\nGenerated {len(sentence_train_pairs):,} Pure Full-Sentence Training Pairs:")
    print(f"  - Speech Audio Anchors: {num_audio:,} ({num_audio/len(sentence_train_pairs):.1%})")
    print(f"  - Native Text Anchors:  {num_text:,} ({num_text/len(sentence_train_pairs):.1%})")

    print(f"\nWriting to {out_train_sentence}...")
    with open(out_train_sentence, "w", encoding="utf-8") as f:
        for p in sentence_train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n" + "=" * 65)
    print("STEP 3: Generating Pure Full-Sentence Out-of-Domain FLEURS Validation Pairs")
    print("=" * 65)

    ood_test_file = data_dir / "ood_test_clips.jsonl"
    out_ood_sentence = data_dir / "ood_val_pairs_sentence.jsonl"

    sentence_ood_pairs = []
    if ood_test_file.exists():
        with open(ood_test_file, "r", encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                lang_name = s["language"]
                clip_p = s["clip_path"]
                native_t = s.get("native_text", "")
                eng_t = s.get("english_text", "")

                if not eng_t or not os.path.exists(clip_p):
                    continue

                # Speech Audio -> Gold English Sentence
                sentence_ood_pairs.append({
                    "anchor": clip_p,
                    "positive": f"{lang_name}: {eng_t}",
                    "modality": "audio",
                })

                # Native Text -> Gold English Sentence
                if native_t:
                    sentence_ood_pairs.append({
                        "anchor": f"{lang_name}: {native_t}",
                        "positive": f"{lang_name}: {eng_t}",
                        "modality": "text",
                    })

    random.seed(42)
    random.shuffle(sentence_ood_pairs)

    with open(out_ood_sentence, "w", encoding="utf-8") as f:
        for p in sentence_ood_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(sentence_ood_pairs):,} Full-Sentence OOD validation pairs to: {out_ood_sentence}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build pure full-sentence foundation dataset.")
    parser.add_argument("--data_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data")
    parser.add_argument("--cache_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data/jw_corpus_cache")
    args = parser.parse_args()
    build_sentence_dataset(args)
