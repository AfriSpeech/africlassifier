"""CPU-First Inference Engine for africlassifier.

Supports both Speech Audio and Native African Text inputs.
Allows user to pick whether to classify into English or French AfricaNouns (default: English).
Optimized for CPU-first execution with optional non-quantized GPU deployment.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from africlassifier.languages import resolve_language

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}


@dataclass
class LabelScore:
    rank: int
    label: str
    type: str
    score: float


@dataclass
class PredictionResult:
    input_type: str  # "audio" or "text"
    input_source: str
    language: Optional[dict]  # {"iso": str, "name": str}
    target_language: str      # "en" or "fr"
    device_used: str          # "cpu" or "cuda"
    threshold: float
    matched_count: int
    inference_time_ms: float
    matched_labels: List[LabelScore]
    top_prediction: Optional[LabelScore]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class AfriClassifier:
    """CPU-first classification foundation engine for African speech and text."""

    def __init__(
        self,
        model,
        indices: Dict[str, Tuple[torch.Tensor, List[dict]]],  # {"en": (tensor, meta), "fr": (tensor, meta)}
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.indices = indices  # Target language indices

    @classmethod
    def load(
        cls,
        model_path: str = "outputs/embeddinggemma_lora",
        indices_dir: str = "data/indices",
        device: str = "cpu",          # Default: CPU-first!
        quantize_cpu: bool = False,   # Optional dynamic int8 CPU quantization
        num_threads: int = 4,         # CPU thread tuning
    ) -> "AfriClassifier":
        """Load the model and AfricaNouns indices (CPU-first by default)."""
        if device == "cpu":
            torch.set_num_threads(num_threads)
            print(f"Loading africlassifier on CPU ({num_threads} threads)...")
        else:
            print(f"Loading africlassifier on GPU ({device}, non-quantized weights)...")

        from unsloth import FastSentenceTransformer

        model = FastSentenceTransformer.from_pretrained(
            model_name=model_path,
            max_seq_length=1024,
            config_kwargs={"vision_config": None},
            full_finetuning=False,
        )
        model.eval()

        if device == "cpu" and quantize_cpu:
            print("Applying dynamic INT8 quantization for CPU inference...")
            try:
                model = torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                print(f"Note: Dynamic quantization skipped ({e}), running standard CPU float.")

        indices_dir_path = Path(indices_dir)
        indices: Dict[str, Tuple[torch.Tensor, List[dict]]] = {}

        # 1. Load English AfricaNouns Index
        en_emb_path = indices_dir_path / "nouns_en_embeddings.pt"
        en_meta_path = indices_dir_path / "nouns_en_metadata.json"
        if en_emb_path.exists() and en_meta_path.exists():
            embs_en = torch.load(en_emb_path, map_location=device).float()
            embs_en = torch.nn.functional.normalize(embs_en, p=2, dim=-1)
            with open(en_meta_path, "r", encoding="utf-8") as f:
                meta_en = json.load(f)
            indices["en"] = (embs_en, meta_en)
            print(f"  ✓ Loaded English AfricaNouns Index ({len(meta_en):,} noun phrases)")

        # 2. Load French AfricaNouns Index
        fr_emb_path = indices_dir_path / "nouns_fr_embeddings.pt"
        fr_meta_path = indices_dir_path / "nouns_fr_metadata.json"
        if fr_emb_path.exists() and fr_meta_path.exists():
            embs_fr = torch.load(fr_emb_path, map_location=device).float()
            embs_fr = torch.nn.functional.normalize(embs_fr, p=2, dim=-1)
            with open(fr_meta_path, "r", encoding="utf-8") as f:
                meta_fr = json.load(f)
            indices["fr"] = (embs_fr, meta_fr)
            print(f"  ✓ Loaded French AfricaNouns Index ({len(meta_fr):,} noun phrases)")

        return cls(model=model, indices=indices, device=device)

    def predict(
        self,
        input_data: Union[str, Path, np.ndarray],  # Audio file path OR African language text string
        language: Optional[str] = None,            # Input language (e.g. 'twi', 'yor', 'swahili')
        target_language: str = "en",               # Noun phrases language: "en" (default) or "fr"
        threshold: float = 0.15,
        top_k: int = 5,
    ) -> PredictionResult:
        """Classify African speech or text into English or French AfricaNouns.

        Args:
            input_data: Audio file path (.wav, .flac) OR African language text string.
            language: Spoken/written language code or name (e.g. 'twi', 'yoruba').
            target_language: Target noun phrases: 'en' (default) or 'fr'.
            threshold: Minimum cosine similarity score.
            top_k: Number of top noun phrases to return.

        Returns:
            PredictionResult: structured result containing matched noun phrases and scores.
        """
        t0 = time.time()
        tgt_lang = target_language.strip().lower()
        if tgt_lang not in self.indices:
            raise ValueError(f"Target language '{target_language}' not available. Choose 'en' or 'fr'.")

        label_matrix, metadata = self.indices[tgt_lang]

        iso_code, lang_name = resolve_language(language)
        input_str = str(input_data).strip()
        is_audio = False

        # Detect modality: Audio file vs. African text string
        if Path(input_str).suffix.lower() in AUDIO_EXTENSIONS and Path(input_str).exists():
            is_audio = True
            input_type = "audio"
            preview_source = input_str
            with torch.no_grad():
                input_emb = self.model.encode(
                    [{"audio": input_str}],
                    batch_size=1,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
        else:
            input_type = "text"
            preview_source = input_str[:120]
            prefix = f"{lang_name}: " if lang_name else ""
            prompt = f"{prefix}{input_str}"
            with torch.no_grad():
                input_emb = self.model.encode(
                    [prompt],
                    batch_size=1,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )

        if self.device != "cpu":
            input_emb = input_emb.to(self.device)
            target_matrix = label_matrix.to(input_emb.dtype)
        else:
            input_emb = input_emb.float().cpu()
            target_matrix = label_matrix.float().cpu()

        # Dot-product Top-K search
        sims = torch.mm(input_emb, target_matrix.T).squeeze(0)
        scores, indices = torch.topk(sims, k=min(top_k, len(metadata)))

        scores = scores.float().cpu().numpy()
        indices = indices.cpu().numpy()

        matched: List[LabelScore] = []
        for rank, (idx, score_val) in enumerate(zip(indices, scores), 1):
            if score_val < threshold and len(matched) > 0:
                break
            matched.append(
                LabelScore(
                    rank=rank,
                    label=metadata[idx]["label"],
                    type="noun_phrase",
                    score=round(float(score_val), 4),
                )
            )

        top_pred = matched[0] if matched else None
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        lang_info = {"iso": iso_code, "name": lang_name} if lang_name else None

        return PredictionResult(
            input_type=input_type,
            input_source=preview_source,
            language=lang_info,
            target_language=tgt_lang,
            device_used=self.device,
            threshold=threshold,
            matched_count=len(matched),
            inference_time_ms=elapsed_ms,
            matched_labels=matched,
            top_prediction=top_pred,
        )
