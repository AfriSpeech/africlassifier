"""Command-line inference tool for speech2topics."""

import argparse
import sys
from pathlib import Path
from afri2labels.classifier import Afri2LabelsClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict English noun phrases and action verbs from African speech audio."
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        required=True,
        help="Path to audio file (.wav, .flac, .mp3, etc.)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="outputs/embeddinggemma_lora",
        help="Path to trained model or checkpoint (default: outputs/embeddinggemma_lora)",
    )
    parser.add_argument(
        "--catalog",
        "-c",
        type=str,
        default="data/codeswitch_nouns_catalog.json",
        help="Path to label catalog JSON file (default: data/codeswitch_nouns_catalog.json from africa-tts)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="data/label_embeddings.pt",
        help="Path to cached label embeddings tensor",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        required=True,
        help="ISO code or language name for conditioning (e.g. 'twi', 'yor', 'hau', 'Twi')",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.35,
        help="Cosine similarity threshold for selecting labels (default: 0.35)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=15,
        help="Maximum number of matched labels to return (default: 15)",
    )
    parser.add_argument(
        "--filter-type",
        choices=["noun_phrase", "noun", "verb"],
        default=None,
        help="Optional filter to only return nouns, noun phrases, or verbs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.audio).exists():
        print(f"Error: Audio file not found at '{args.audio}'", file=sys.stderr)
        sys.exit(1)

    classifier = Afri2LabelsClassifier.load(
        model_path=args.model,
        catalog_path=args.catalog,
        embeddings_cache_path=args.cache,
        device=args.device,
    )

    result = classifier.predict(
        input_data=args.audio,
        language=args.language,
        threshold=args.threshold,
        top_k=args.top_k,
        filter_type=args.filter_type,
    )

    # Print structured JSON directly to stdout
    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()
