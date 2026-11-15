#!/usr/bin/env python3
"""Fine-tune EmbeddingGemma 2 with LoRA for African Speech -> English Concepts.

Based on Unsloth EmbeddingGemma 2 multimodal Conformer audio training.
Uses MultipleNegativesRankingLoss with large in-batch negatives on NVIDIA H200.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from unsloth import FastSentenceTransformer, is_bf16_supported
from speech2topics.languages import resolve_language


def load_pairs_dataset(
    jsonl_path: str,
    max_samples: int = None,
    lang_prefix_ratio: float = 1.0,
) -> Dataset:
    """Load JSONL contrastive pairs: {"anchor": audio_path, "positive": label}.
    
    Uses 100% full language conditioning (e.g. "Twi: egg", "Yoruba: harvest")
    for clean contrastive separation and complete cross-lingual disambiguation.
    """
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            audio_path = r["anchor"]
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                continue

            label = r["positive"]
            raw_lang = r.get("language") or r.get("subset", "").split("_")[-1]
            iso_code, lang_name = resolve_language(raw_lang)

            # Full natural language conditioning: "Twi: egg", "Yoruba: harvest"
            if lang_name:
                target_text = f"{lang_name}: {label}"
            else:
                target_text = label

            pairs.append({
                "anchor": {"audio": audio_path},
                "positive": target_text,
            })
            if max_samples and len(pairs) >= max_samples:
                break
    random.seed(42)
    random.shuffle(pairs)
    sample_pref = pairs[0]['positive'] if pairs else ''
    print(f"Loaded {len(pairs):,} verified pairs from {jsonl_path} (100% language-conditioned, sample: '{sample_pref}')")
    return Dataset.from_list(pairs)


def train(args):
    # Set Hopper TF32 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"STEP 1: Loading Base Multimodal EmbeddingGemma 2 Model")
    print(f"Base model: {args.model_name}")
    print("=" * 65)

    model = FastSentenceTransformer.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        config_kwargs={"vision_config": None},  # Strip unused vision tower
        full_finetuning=False,
    )

    print("\n" + "=" * 65)
    print(f"STEP 2: Adding LoRA Adapters to Language & Audio Conformer Layers")
    print(f"LoRA Rank: r={args.lora_r}, Alpha={args.lora_alpha}")
    print("=" * 65)

    model = FastSentenceTransformer.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        finetune_audio_layers=True,  # Fine-tune the audio Conformer encoder
        lora_alpha=args.lora_alpha,
        lora_dropout=0,              # Optimized in Unsloth
        bias="none",
        use_gradient_checkpointing=args.gradient_checkpointing,
        random_state=42,
        task_type="FEATURE_EXTRACTION",
    )

    print("\n" + "=" * 65)
    print(f"STEP 3: Loading Datasets")
    print("=" * 65)
    train_dataset = load_pairs_dataset(args.train_file, max_samples=args.max_train_samples)
    val_dataset = load_pairs_dataset(args.val_file, max_samples=args.max_val_samples) if args.val_file else None

    # MultipleNegativesRankingLoss with in-batch negatives
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.save_steps if val_dataset else None,
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
        tf32=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss_fn,
    )

    # Print GPU memory before training
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_props.name} | Total VRAM: {gpu_props.total_memory / 1e9:.1f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print("\n" + "=" * 65)
    print(f"STEP 4: Launching Training on H200")
    print("=" * 65)

    stats = trainer.train()

    print("\n" + "=" * 65)
    print(f"STEP 5: Saving Fine-Tuned Model & Adapters")
    print("=" * 65)
    save_path = output_dir / "embeddinggemma_lora"
    model.save_pretrained(str(save_path))
    model.tokenizer.save_pretrained(str(save_path))
    print(f"LoRA model saved to: {save_path}")

    if args.merge_16bit:
        merged_path = output_dir / "embeddinggemma_merged_16bit"
        print(f"Merging weights and saving full 16-bit model to: {merged_path}...")
        model.save_pretrained_merged(
            str(merged_path),
            tokenizer=model.tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model saved to: {merged_path}")

    print("\nTraining completed successfully!")
    print(f"Runtime: {stats.metrics.get('train_runtime', 0):.1f}s")
    print(f"Peak Reserved Memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune EmbeddingGemma 2 with LoRA on African Speech to Concepts.")
    parser.add_argument("--train_file", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data/train_pairs.jsonl")
    parser.add_argument("--val_file", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/data/ood_val_pairs.jsonl", help="Out-of-domain non-religious validation file")
    parser.add_argument("--output_dir", type=str, default="/mnt/volume_d2wey28/projects/speech2topics/outputs")
    parser.add_argument("--model_name", type=str, default="unsloth/embeddinggemma-2")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size (16 fits comfortably alongside heavy background jobs)")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Effective batch size 64 (16x4)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=64, help="High-Rank LoRA (default 64)")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA Alpha (default 128)")
    parser.add_argument("--gradient_checkpointing", type=str, default="unsloth")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--merge_16bit", action="store_true", help="Merge LoRA weights into standalone 16-bit model")

    args = parser.parse_args()
    train(args)
