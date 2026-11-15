#!/usr/bin/env python3
"""Build Out-of-Domain (OOD) Non-Religious African Speech Evaluation Dataset.

Sources:
- Google FLEURS African subsets (FLoRes-101 domain: news, science, history, geography, daily life).
- 100% Non-Religious ground truth text.
- Aligned by sentence ID to gold English reference sentences.
- Concepts extracted strictly via spaCy doc.noun_chunks and action verbs.
- Full language conditioning: "<Language>: <Concept>" (e.g. "Swahili: dinosaurs").
"""

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import soxr
from datasets import Audio, load_dataset
from tqdm import tqdm

from speech2topics.extractor import extract_concept_labels
from speech2topics.languages import resolve_language

FLEURS_AFRICAN_CONFIGS = [
    ("sw_ke", "Swahili", "swa"),
    ("yo_ng", "Yoruba", "yor"),
    ("ha_ng", "Hausa", "hau"),
    ("am_et", "Amharic", "amh"),
    ("zu_za", "Zulu", "zul"),
    ("ig_ng", "Igbo", "ibo"),
    ("ln_cd", "Lingala", "lin"),
    ("om_et", "Oromo", "orm"),
    ("so_so", "Somali", "som"),
    ("wo_sn", "Wolof", "wol"),
    ("lg_ug", "Ganda", "lug"),
    ("sn_zw", "Shona", "sna"),
    ("nso_za", "Northern Sotho", "nso"),
    ("af_za", "Afrikaans", "afr"),
]


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


def build_ood_dataset(args):
    data_dir = Path(args.data_dir)
    clips_ood_dir = data_dir / "clips_ood"
    clips_ood_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STEP 1: Indexing Gold English Non-Religious References (FLEURS en_us)")
    print("=" * 65)

    eng_concepts_map: Dict[int, dict] = {}

    for split in ["validation", "test"]:
        print(f"Loading FLEURS en_us {split} split...")
        ds_en = load_dataset("google/fleurs", "en_us", split=split, streaming=True)
        ds_en = ds_en.cast_column("audio", Audio(decode=False))

        for row in ds_en:
            sid = row["id"]
            text = row["transcription"]
            concepts = extract_concept_labels(text)
            eng_concepts_map[sid] = {
                "text": text,
                "noun_phrases": concepts["noun_phrases"],
                "verbs": concepts["verbs"],
                "all_labels": concepts["all_labels"],
            }

    print(f"Indexed {len(eng_concepts_map):,} gold English non-religious sentences.")

    print("\n" + "=" * 65)
    print("STEP 2: Processing 14 African Languages (FLEURS Non-Religious Audio)")
    print("=" * 65)

    ood_val_samples = []
    ood_test_samples = []

    for cfg, lang_name, iso in FLEURS_AFRICAN_CONFIGS:
        print(f"\nProcessing {lang_name} ({cfg}, {iso})...")
        for split, sample_list in [("validation", ood_val_samples), ("test", ood_test_samples)]:
            try:
                ds = load_dataset("google/fleurs", cfg, split=split, streaming=True)
                ds = ds.cast_column("audio", Audio(decode=False))
            except Exception as e:
                print(f"  Warning: Could not load {cfg} {split}: {e}")
                continue

            split_clips = 0
            for row in ds:
                sid = row["id"]
                if sid not in eng_concepts_map:
                    continue

                eng_info = eng_concepts_map[sid]
                all_labels = eng_info["all_labels"]
                if not all_labels:
                    continue

                clip_filename = f"fleurs_{iso}_{split}_{sid}.flac"
                dst_audio = clips_ood_dir / clip_filename

                audio_bytes = row["audio"]["bytes"]
                if not dst_audio.exists():
                    write_flac(audio_bytes, str(dst_audio))

                sample = {
                    "clip_path": str(dst_audio),
                    "id": sid,
                    "split": split,
                    "subset": f"fleurs_{cfg}",
                    "language": lang_name,
                    "iso": iso,
                    "domain": "fleurs_non_religious",
                    "native_text": row["transcription"],
                    "english_text": eng_info["text"],
                    "noun_phrases": [n["label"] for n in eng_info["noun_phrases"]],
                    "verbs": [v["label"] for v in eng_info["verbs"]],
                    "all_labels": [l["label"] for l in all_labels],
                    "label_objects": all_labels,
                }
                sample_list.append(sample)
                split_clips += 1
                if args.max_clips_per_lang and split_clips >= args.max_clips_per_lang:
                    break

            print(f"  ✓ {lang_name} {split:10s}: {split_clips:4d} clips aligned to English concepts")

    print("\n" + "=" * 65)
    print("STEP 3: Generating Contrastive OOD Validation Pairs")
    print("=" * 65)

    # Convert val samples to pairs for SentenceTransformerTrainer evaluation
    ood_val_pairs = []
    for s in ood_val_samples:
        lang_name = s["language"]
        for item in s["label_objects"]:
            ood_val_pairs.append({
                "anchor": s["clip_path"],
                "positive": f"{lang_name}: {item['label']}",
                "label_type": item["type"],
                "language": s["iso"],
            })

    # Save to disk
    with open(data_dir / "ood_val_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in ood_val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(data_dir / "ood_test_clips.jsonl", "w", encoding="utf-8") as f:
        for s in ood_test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved {len(ood_val_pairs):,} OOD validation pairs to: {data_dir / 'ood_val_pairs.jsonl'}")
    print(f"Saved {len(ood_test_samples):,} OOD test clips to: {data_dir / 'ood_test_clips.jsonl'}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Out-Of-Domain Non-Religious African Speech Eval Dataset.")
    parser.add_argument("--data_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data")
    parser.add_argument("--max_clips_per_lang", type=int, default=None, help="Max clips per language (default None = all)")
    args = parser.parse_args()
    build_ood_dataset(args)
