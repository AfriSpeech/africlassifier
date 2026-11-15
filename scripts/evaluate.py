#!/usr/bin/env python3
"""Evaluate Speech-to-Topics retrieval: Recall@1, Recall@5, Recall@10, and MRR."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
from tqdm import tqdm
from unsloth import FastSentenceTransformer


def recall_and_mrr(
    sim_matrix: torch.Tensor,  # [num_queries, num_candidates]
    positives_map: List[Set[int]],  # ground truth candidate indices per query
    ks=(1, 5, 10),
) -> dict:
    """Compute Recall@K and Mean Reciprocal Rank."""
    ranking = sim_matrix.argsort(dim=1, descending=True)
    results = {}
    
    for k in ks:
        hits = 0
        for q, top_k in enumerate(ranking[:, :k].tolist()):
            if set(top_k) & positives_map[q]:
                hits += 1
        results[f"R@{k}"] = round(100.0 * hits / len(positives_map), 2)

    # MRR
    reciprocal_ranks = []
    for q, ranked in enumerate(ranking.tolist()):
        found_rank = 0
        for rank_idx, cand_idx in enumerate(ranked, 1):
            if cand_idx in positives_map[q]:
                found_rank = 1.0 / rank_idx
                break
        reciprocal_ranks.append(found_rank)

    results["MRR"] = round(100.0 * float(np.mean(reciprocal_ranks)), 2)
    return results


def evaluate(args):
    print("=" * 65)
    print("Evaluating Speech2Topics Model")
    print(f"Model path: {args.model_path}")
    print(f"Test file:  {args.test_file}")
    print("=" * 65)

    model = FastSentenceTransformer.from_pretrained(
        model_name=args.model_path,
        max_seq_length=1024,
        config_kwargs={"vision_config": None},
        full_finetuning=False,
    )
    model.eval()

    # Load test clips
    test_clips = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            test_clips.append(json.loads(line))
            if args.max_clips and len(test_clips) >= args.max_clips:
                break

    print(f"Loaded {len(test_clips):,} test audio clips.")

    # Collect candidate pool of labels
    all_candidate_labels = sorted(list({lbl for s in test_clips for lbl in s["all_labels"]}))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(all_candidate_labels)}
    print(f"Candidate label vocabulary: {len(all_candidate_labels):,} concepts.")

    # Encode all candidate labels
    with torch.no_grad():
        print("Encoding candidate labels...")
        label_embeddings = model.encode(
            all_candidate_labels,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # Encode test audio clips
        print("Encoding test audio clips...")
        audio_inputs = [{"audio": s["clip_path"]} for s in test_clips]
        audio_embeddings = model.encode(
            audio_inputs,
            batch_size=args.audio_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # Cosine similarity matrix: [num_audio, num_labels]
        sims = torch.mm(audio_embeddings, label_embeddings.T).float().cpu()

    # Ground truth positives for each audio clip
    positives = []
    for s in test_clips:
        pos_indices = {label_to_idx[lbl] for lbl in s["all_labels"] if lbl in label_to_idx}
        positives.append(pos_indices)

    metrics = recall_and_mrr(sims, positives, ks=(1, 5, 10))

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    for k, v in metrics.items():
        print(f"  {k:8s}: {v:6.2f}%")
    print("=" * 65)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Speech2Topics model.")
    parser.add_argument("--model_path", type=str, default="outputs/embeddinggemma_lora")
    parser.add_argument("--test_file", type=str, default="data/test_clips.jsonl")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--audio_batch_size", type=int, default=32)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)

    args = parser.parse_args()
    evaluate(args)
