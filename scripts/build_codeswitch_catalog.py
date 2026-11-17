#!/usr/bin/env python3
"""Build Inference Catalog using ONLY the English noun phrases & entities bundled with africa-tts.

Strict Constraints:
- No training data labels (no JW / Bible concepts).
- No verbs.
- 100% pure English noun phrases and African named entities from africa-tts codeswitching lexicon.
"""

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path


def build_codeswitch_catalog(args):
    data_dir = Path(args.data_dir)
    tsv_path = Path(args.tsv_path)
    out_path = data_dir / "codeswitch_nouns_catalog.json"

    print("=" * 65)
    print("Building africa-tts Code-Switching Noun Phrases Catalog")
    print(f"Source Lexicon: {tsv_path}")
    print(f"Output File:    {out_path}")
    print("=" * 65)

    if not tsv_path.exists():
        print(f"Error: {tsv_path} not found!", file=sys.stderr)
        sys.exit(1)

    seen = set()
    catalog = []

    open_fn = gzip.open if str(tsv_path).endswith(".gz") else open
    with open_fn(tsv_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            phrase = parts[0].strip().lower()

            # Filter: clean phrase, minimum length, no pure numbers
            phrase = re.sub(r"\s+", " ", phrase)
            if len(phrase) < 3 or phrase.isdigit():
                continue

            if phrase not in seen:
                seen.add(phrase)
                catalog.append({
                    "label": phrase,
                    "type": "noun_phrase",
                    "source": "africa_tts_codeswitch",
                })
                if args.limit and len(catalog) >= args.limit:
                    break

    print(f"\nExtracted {len(catalog):,} unique English noun phrases & named entities from africa-tts.")
    print("Sample phrases (first 10):")
    for item in catalog[:10]:
        print("  -", item["label"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nSaved catalog to: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract africa-tts codeswitch noun phrases.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--tsv_path", type=str, default="data/codeswitch_noun_phrases.tsv.gz")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit (default None = all)")
    args = parser.parse_args()
    build_codeswitch_catalog(args)
