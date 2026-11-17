"""Command-line multi-task classification tool for africlassifier (CPU-first by default)."""

import argparse
import sys
from pathlib import Path
from africlassifier.classifier import AfriClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="africlassifier: Zero-shot Multi-Task Classification (Domain, Sentiment, Emotion, Intent, Language)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--audio",
        "-a",
        type=str,
        help="Path to audio file (.wav, .flac, .mp3, etc.)",
    )
    group.add_argument(
        "--text",
        type=str,
        help="African language text string to classify",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Optional language hint (e.g. 'twi', 'yor', 'swahili'). Auto-detected if None.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=1,
        help="Number of top predictions per task (default: 1 for clean top-1 output)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="outputs/embeddinggemma_lora",
        help="Path to trained model or checkpoint (default: outputs/embeddinggemma_lora)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data/taxonomies_index.pt",
        help="Path to precomputed taxonomy index (default: data/taxonomies_index.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device (default: 'cpu' for fast lightweight inference, or 'cuda')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_data = args.audio if args.audio else args.text
    if args.audio and not Path(args.audio).exists():
        print(f"Error: Audio file not found at '{args.audio}'", file=sys.stderr)
        sys.exit(1)

    classifier = AfriClassifier.load(
        model_path=args.model,
        taxonomy_index_path=args.index,
        device=args.device,
    )

    result = classifier.predict(
        input_data=input_data,
        language=args.language,
        top_k=args.top_k,
    )

    # Print clean structured JSON
    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()
