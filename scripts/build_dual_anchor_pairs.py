#!/usr/bin/env python3
"""Build Dual-Anchor (Audio + African Text -> English Concepts) dataset for afri2labels.

Generates:
1. Audio Contrastive Pairs: {"anchor": {"audio": clip_path}, "positive": f"{Lang}: {Concept}"}
2. Text Contrastive Pairs:  {"anchor": f"{Lang}: {African_Text}", "positive": f"{Lang}: {Concept}"}
"""

import argparse
import glob
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


def build_dual_anchor_pairs(args):
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    print("=" * 65)
    print("STEP 1: Indexing African Language Text from Parallel Corpus")
    print("=" * 65)

    csv_files = [f for f in os.listdir(cache_dir) if f.endswith(".csv") and f != "english_cache.csv"]
    iso_vk_to_text: Dict[Tuple[str, str], str] = {}

    t0 = time.time()
    for f in tqdm(csv_files, desc="Indexing CSVs"):
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

    print(f"Indexed {len(iso_vk_to_text):,} (code, verse_key) -> African text entries in {time.time()-t0:.1f}s.")

    print("\n" + "=" * 65)
    print("STEP 2: Building Dual-Anchor Training Pairs (Speech + African Text)")
    print("=" * 65)

    train_file = data_dir / "train_pairs.jsonl"
    out_train_dual = data_dir / "train_pairs_dual.jsonl"

    audio_pairs = []
    text_pairs_map: Dict[Tuple[str, str, str], dict] = {}

    with open(train_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading Audio Pairs"):
            r = json.loads(line)
            audio_path = r["anchor"]
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                continue

            iso = r.get("language") or r.get("subset", "").split("_")[-1]
            vk = str(r.get("verse_key", ""))
            label = r["positive"]
            _, lang_name = resolve_language(iso)
            prefix = lang_name or "African"

            # 1. Audio pair (audio file path string)
            audio_pairs.append({
                "anchor": audio_path,
                "positive": f"{prefix}: {label}",
                "modality": "audio",
            })

            # 2. Text pair (African text string)
            alt_code = ISO3_TO_ISO1.get(iso.lower(), iso.lower())
            lookup_key = (iso.lower(), vk) if (iso.lower(), vk) in iso_vk_to_text else (alt_code, vk)
            
            if lookup_key in iso_vk_to_text:
                afri_text = iso_vk_to_text[lookup_key]
                t_key = (prefix, afri_text[:60], label)
                if t_key not in text_pairs_map:
                    text_pairs_map[t_key] = {
                        "anchor": f"{prefix}: {afri_text}",
                        "positive": f"{prefix}: {label}",
                        "modality": "text",
                    }

    text_pairs = list(text_pairs_map.values())
    print(f"Total Audio Pairs: {len(audio_pairs):,}")
    print(f"Total African Text Pairs: {len(text_pairs):,}")

    # Combine and shuffle
    all_dual_train = audio_pairs + text_pairs
    random.seed(42)
    random.shuffle(all_dual_train)

    print(f"Writing {len(all_dual_train):,} dual-anchor training pairs to {out_train_dual}...")
    with open(out_train_dual, "w", encoding="utf-8") as f:
        for p in all_dual_train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n" + "=" * 65)
    print("STEP 3: Building Dual-Anchor Out-of-Domain Validation Pairs")
    print("=" * 65)

    ood_val_file = data_dir / "ood_val_pairs.jsonl"
    out_ood_dual = data_dir / "ood_val_pairs_dual.jsonl"
    test_clips_file = data_dir / "ood_test_clips.jsonl"

    ood_dual_pairs = []
    with open(ood_val_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            audio_p = r["anchor"]
            if isinstance(audio_p, dict):
                audio_p = audio_p.get("audio")
            ood_dual_pairs.append({
                "anchor": audio_p,
                "positive": r["positive"],
                "modality": "audio",
            })

    # Add African text OOD validation pairs from test clips
    if test_clips_file.exists():
        with open(test_clips_file, "r", encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                lang_name = s["language"]
                native_text = s.get("native_text", "")
                if not native_text:
                    continue
                for item in s["label_objects"]:
                    ood_dual_pairs.append({
                        "anchor": f"{lang_name}: {native_text}",
                        "positive": f"{lang_name}: {item['label']}",
                        "modality": "text",
                    })

    random.seed(42)
    random.shuffle(ood_dual_pairs)
    with open(out_ood_dual, "w", encoding="utf-8") as f:
        for p in ood_dual_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(ood_dual_pairs):,} dual-anchor OOD validation pairs to: {out_ood_dual}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dual-anchor (speech + African text) training dataset.")
    parser.add_argument("--data_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data")
    parser.add_argument("--cache_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data/jw_corpus_cache")
    args = parser.parse_args()
    build_dual_anchor_pairs(args)
