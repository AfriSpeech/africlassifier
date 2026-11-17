"""Command-line inference tool for africlassifier (CPU-First by default)."""

import argparse
import sys
from pathlib import Path
from africlassifier.classifier import AfriClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="africlassifier: Classify African speech audio OR text into English or French AfricaNouns."
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
        help="Input language code or name (e.g. 'twi', 'yor', 'swahili', 'amharic')",
    )
    parser.add_argument(
        "--target-lang",
        "-t",
        choices=["en", "fr"],
        default="en",
        help="Target noun phrases language: 'en' (default, English) or 'fr' (French)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="outputs/embeddinggemma_lora",
        help="Path to trained model or checkpoint (default: outputs/embeddinggemma_lora)",
    )
    parser.add_argument(
        "--indices-dir",
        type=str,
        default="data/indices",
        help="Directory containing precomputed AfricaNouns indices (default: data/indices)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device (default: 'cpu' for fast lightweight inference, or 'cuda')",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable dynamic INT8 quantization for CPU",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Cosine similarity threshold for selecting labels (default: 0.15)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Maximum number of matched labels to return (default: 5)",
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
        indices_dir=args.indices_dir,
        device=args.device,
        quantize_cpu=args.quantize,
    )

    result = classifier.predict(
        input_data=input_data,
        language=args.language,
        target_language=args.target_lang,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    # Print clean structured JSON
    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()
