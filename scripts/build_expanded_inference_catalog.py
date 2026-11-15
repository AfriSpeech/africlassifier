#!/usr/bin/env python3
"""Build expanded inference catalog incorporating AfricaNouns, Named Entities, and Action Verbs.

This allows zero-shot generalization at inference time:
- Top African Nouns & Concepts (from AfricaNouns)
- African Named Entities (from Africa-TTS codeswitch lexicon)
- Core Action Verbs (from trained catalog)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm


def build_expanded_catalog(args):
    data_dir = Path(args.data_dir)
    out_path = data_dir / "expanded_labels.json"

    print("=" * 65)
    print("Building Expanded Inference Concept & Named Entity Catalog")
    print("=" * 65)

    catalog: Dict[str, dict] = {}

    # 1. Start with our trained action verbs and core concepts
    trained_catalog_path = data_dir / "trained_labels.json"
    if trained_catalog_path.exists():
        print(f"Loading base trained catalog from {trained_catalog_path}...")
        with open(trained_catalog_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                lbl = item["label"].strip().lower()
                catalog[lbl] = {
                    "label": lbl,
                    "type": item["type"],
                    "source": "trained_base",
                    "priority": 1,
                }
        num_verbs = sum(1 for v in catalog.values() if v["type"] == "verb")
        num_nps = sum(1 for v in catalog.values() if v["type"] == "noun_phrase")
        print(f"  ✓ Loaded {len(catalog):,} base concepts ({num_verbs:,} verbs, {num_nps:,} noun phrases)")

    # 2. Add ALL high-quality African Nouns from AfricaNouns parquet
    africa_nouns_path = Path(args.africa_nouns_path)
    if africa_nouns_path.exists():
        print(f"\nLoading ALL AfricaNouns from {africa_nouns_path}...")
        df_nouns = pd.read_parquet(africa_nouns_path)
        # Filter: minimum count, minimum length, spread across at least 2 countries
        filtered_nouns = df_nouns[
            (df_nouns["total_count"] >= args.min_noun_count)
            & (df_nouns["phrase"].str.len() >= 3)
            & (df_nouns["country_spread"] >= args.min_country_spread)
        ]
        print(f"  Filtered {len(filtered_nouns):,} high-quality African nouns.")

        added_nouns = 0
        noun_iterable = filtered_nouns if not args.max_africa_nouns else filtered_nouns.head(args.max_africa_nouns)
        for _, row in noun_iterable.iterrows():
            phrase = str(row["phrase"]).strip().lower()
            if phrase not in catalog and not phrase.isdigit():
                catalog[phrase] = {
                    "label": phrase,
                    "type": "noun_phrase",
                    "source": "africa_nouns",
                    "priority": 2,
                }
                added_nouns += 1
        print(f"  ✓ Added {added_nouns:,} African noun phrases.")

    # 3. Add ALL African Named Entities from codeswitch lexicons
    codeswitch_path = Path(args.codeswitch_path)
    if codeswitch_path.exists():
        print(f"\nLoading ALL African Named Entities from {codeswitch_path}...")
        with open(codeswitch_path, "r", encoding="utf-8") as f:
            cs_data = json.load(f)
        added_entities = 0
        for term in cs_data.keys():
            clean_term = term.strip().lower()
            if len(clean_term) >= 3 and clean_term not in catalog and not clean_term.isdigit():
                catalog[clean_term] = {
                    "label": clean_term,
                    "type": "named_entity",
                    "source": "africa_named_entities",
                    "priority": 3,
                }
                added_entities += 1
                if args.max_named_entities and added_entities >= args.max_named_entities:
                    break
        print(f"  ✓ Added {added_entities:,} African named entities.")

    # Finalize catalog list sorted by priority and label length
    final_list = sorted(catalog.values(), key=lambda x: (x["priority"], x["label"]))

    print("\n" + "=" * 65)
    print("EXPANDED INFERENCE CATALOG SUMMARY")
    print("=" * 65)
    print(f"Total concepts & entities in catalog: {len(final_list):,}")
    print(f"  - Base trained concepts:            {sum(1 for x in final_list if x['source'] == 'trained_base'):,}")
    print(f"  - AfricaNouns phrases:              {sum(1 for x in final_list if x['source'] == 'africa_nouns'):,}")
    print(f"  - African Named Entities:           {sum(1 for x in final_list if x['source'] == 'africa_named_entities'):,}")
    print(f"Saving catalog to: {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build expanded inference catalog with AfricaNouns and Named Entities.")
    parser.add_argument("--data_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data")
    parser.add_argument("--africa_nouns_path", type=str, default="/mnt/volume_d2wey28/projects/africa-nouns/data/africa-nouns-en.parquet")
    parser.add_argument("--codeswitch_path", type=str, default="/mnt/volume_d2wey28/projects/codeswitch_en_core.json")
    parser.add_argument("--min_noun_count", type=int, default=10, help="Min count threshold for AfricaNouns")
    parser.add_argument("--min_country_spread", type=int, default=1, help="Min countries spread across")
    parser.add_argument("--max_africa_nouns", type=int, default=None, help="Max nouns (None = use ALL)")
    parser.add_argument("--max_named_entities", type=int, default=None, help="Max entities (None = use ALL)")
    args = parser.parse_args()
    build_expanded_catalog(args)
