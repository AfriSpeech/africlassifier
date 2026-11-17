"""CPU-First Multi-Task Classification Engine for africlassifier.

Performs zero-shot classification across 5 core NLP dimensions:
1. Language Identification (100+ African languages + English + French)
2. Domain / Topic Classification
3. Sentiment Analysis
4. Emotion Recognition
5. Intent Detection (Customer Support & Chatbots)
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

from africlassifier.languages import NAME_TO_ISO, resolve_language
from africlassifier.taxonomies import DOMAINS, EMOTIONS, INTENTS, LANGUAGES, SENTIMENTS

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}


@dataclass
class ScoredClass:
    label: str
    score: float
    iso: Optional[str] = None  # Populated for language task


@dataclass
class TaskPrediction:
    top_1: ScoredClass
    top_k: List[ScoredClass]


@dataclass
class MultiTaskResult:
    input_type: str        # "audio" or "text"
    input_source: str      # file path or text snippet
    duration_seconds: Optional[float]
    device_used: str       # "cpu" or "cuda"
    inference_time_ms: float
    detected_language: ScoredClass
    language: TaskPrediction
    domain: TaskPrediction
    sentiment: TaskPrediction
    emotion: TaskPrediction
    intent: TaskPrediction
    safety: TaskPrediction

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class AfriClassifier:
    """CPU-first multi-task foundation classifier for African speech and text."""

    def __init__(
        self,
        model,
        taxonomies: Dict[str, dict],  # {"domain": {"keys": [...], "embeddings": tensor}, ...}
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.taxonomies = taxonomies

    @classmethod
    def load(
        cls,
        model_path: str = "outputs/embeddinggemma_lora",
        taxonomy_index_path: str = "data/taxonomies_index.pt",
        device: str = "cpu",          # Default: CPU-first!
        num_threads: int = 4,         # CPU thread optimization
    ) -> "AfriClassifier":
        """Load model and multi-task taxonomy index (CPU-first by default)."""
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

        # Load precomputed taxonomy index
        idx_path = Path(taxonomy_index_path)
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Taxonomy index '{idx_path}' not found! "
                "Run 'python scripts/precompute_taxonomies.py' first."
            )

        raw_index = torch.load(idx_path, map_location=device)
        taxonomies = {}
        for task_name, task_data in raw_index.items():
            embs = task_data["embeddings"].float()
            # Normalize for cosine similarity
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
            taxonomies[task_name] = {
                "keys": task_data["keys"],
                "embeddings": embs.to(device),
            }

        print(f"  ✓ Multi-Task Taxonomy Loaded: {', '.join(taxonomies.keys())}")
        return cls(model=model, taxonomies=taxonomies, device=device)

    def _score_task(
        self,
        query_emb: torch.Tensor,
        task_name: str,
        top_k: int = 1,
    ) -> TaskPrediction:
        """Compute cosine similarity and softmax confidence scores for a task."""
        task_data = self.taxonomies[task_name]
        keys = task_data["keys"]
        matrix = task_data["embeddings"].to(query_emb.dtype)

        sims = torch.mm(query_emb, matrix.T).squeeze(0)  # [num_classes]
        scores, indices = torch.topk(sims, k=min(max(top_k, 1), len(keys)))

        scores = scores.float().cpu().numpy()
        indices = indices.cpu().numpy()

        ranked = []
        for rank, (idx, score_val) in enumerate(zip(indices, scores), 1):
            key = keys[idx]
            iso = NAME_TO_ISO.get(key.lower()) if task_name == "language" else None
            ranked.append(
                ScoredClass(
                    label=key,
                    score=round(float(score_val), 4),
                    iso=iso,
                )
            )

        return TaskPrediction(
            top_1=ranked[0],
            top_k=ranked,
        )

    def predict(
        self,
        input_data: Union[str, Path, np.ndarray],  # Audio file path OR African language text string
        language: Optional[str] = None,            # Optional language hint (auto-detected if None)
        top_k: int = 1,                            # Default top-1 prediction per task
    ) -> MultiTaskResult:
        """Classify input across Domain, Emotion, Sentiment, Intent, and Language.

        Args:
            input_data: Audio file path (.wav, .flac) OR African language text string.
            language: Optional language code/name. If None, language is auto-detected.
            top_k: Number of ranked options to return (default: 1 for concise top-1).

        Returns:
            MultiTaskResult: structured result containing top predictions for all 5 tasks.
        """
        t0 = time.time()
        input_str = str(input_data).strip()
        is_audio = False
        duration = None

        # 1. Detect input modality: Audio vs. Text
        if Path(input_str).suffix.lower() in AUDIO_EXTENSIONS and Path(input_str).exists():
            is_audio = True
            input_type = "audio"
            preview_source = input_str
            try:
                info = sf.info(input_str)
                duration = round(info.duration, 2)
            except Exception:
                duration = 0.0

            with torch.no_grad():
                q_emb = self.model.encode(
                    [{"audio": input_str}],
                    batch_size=1,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
        else:
            input_type = "text"
            preview_source = input_str[:120]
            _, lang_name = resolve_language(language)
            prefix = f"{lang_name}: " if lang_name else ""
            prompt = f"{prefix}{input_str}"

            with torch.no_grad():
                q_emb = self.model.encode(
                    [prompt],
                    batch_size=1,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )

        if self.device != "cpu":
            q_emb = q_emb.to(self.device)
        else:
            q_emb = q_emb.float().cpu()

        # 2. Score all 6 classification tasks
        lang_pred = self._score_task(q_emb, "language", top_k=top_k)
        dom_pred = self._score_task(q_emb, "domain", top_k=top_k)
        sent_pred = self._score_task(q_emb, "sentiment", top_k=top_k)
        emo_pred = self._score_task(q_emb, "emotion", top_k=top_k)
        int_pred = self._score_task(q_emb, "intent", top_k=top_k)
        safe_pred = self._score_task(q_emb, "safety", top_k=top_k)

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        return MultiTaskResult(
            input_type=input_type,
            input_source=preview_source,
            duration_seconds=duration,
            device_used=self.device,
            inference_time_ms=elapsed_ms,
            detected_language=lang_pred.top_1,
            language=lang_pred,
            domain=dom_pred,
            sentiment=sent_pred,
            emotion=emo_pred,
            intent=int_pred,
            safety=safe_pred,
        )
