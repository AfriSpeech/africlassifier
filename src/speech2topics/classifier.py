"""Inference engine for speech-to-topics/concepts prediction."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
import soundfile as sf


@dataclass
class LabelScore:
    label: str
    type: str  # "noun_phrase", "noun", "verb"
    score: float


from speech2topics.languages import resolve_language


@dataclass
class PredictionResult:
    audio: str
    duration_seconds: float
    threshold: float
    language_conditioned: Optional[dict]  # {"iso": str, "name": str} or None
    matched_count: int
    matched_labels: List[LabelScore]
    top_prediction: Optional[LabelScore]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class SpeechTopicClassifier:
    """Classifies audio in African languages directly to English noun phrases & action verbs."""

    def __init__(
        self,
        model,
        label_catalog: List[dict],
        label_embeddings: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.label_catalog = label_catalog  # list of {"label": str, "type": str}
        self.label_texts = [item["label"] for item in label_catalog]
        self.label_types = [item["type"] for item in label_catalog]
        
        self.lang_embeddings_cache: dict[str, torch.Tensor] = {}

        if label_embeddings is not None:
            self.label_embeddings = label_embeddings.to(device)
        else:
            self.label_embeddings = None

    @classmethod
    def load(
        cls,
        model_path: str,
        catalog_path: str,
        embeddings_cache_path: Optional[str] = None,
        device: str = "cuda",
    ) -> "SpeechTopicClassifier":
        """Load fine-tuned model and label catalog from disk."""
        from unsloth import FastSentenceTransformer

        model = FastSentenceTransformer.from_pretrained(
            model_name=model_path,
            max_seq_length=1024,
            config_kwargs={"vision_config": None},
            full_finetuning=False,
        )
        model.eval()

        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        label_embeddings = None
        if embeddings_cache_path and Path(embeddings_cache_path).exists():
            label_embeddings = torch.load(embeddings_cache_path, map_location=device)

        classifier = cls(
            model=model,
            label_catalog=catalog,
            label_embeddings=label_embeddings,
            device=device,
        )

        # Precompute label embeddings if not loaded from cache
        if classifier.label_embeddings is None:
            classifier.precompute_label_embeddings(cache_path=embeddings_cache_path)

        return classifier

    def precompute_label_embeddings(
        self,
        batch_size: int = 256,
        cache_path: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode all candidate labels in catalog once and normalize."""
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(
                self.label_texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        self.label_embeddings = embeddings.to(self.device)
        if cache_path:
            torch.save(self.label_embeddings.cpu(), cache_path)
        return self.label_embeddings

    def get_label_embeddings(self, lang_name: Optional[str] = None) -> torch.Tensor:
        """Retrieve precomputed embeddings for unconditioned or language-conditioned labels."""
        if not lang_name:
            if self.label_embeddings is None:
                self.precompute_label_embeddings()
            return self.label_embeddings

        cache_key = lang_name.strip().lower()
        if cache_key in self.lang_embeddings_cache:
            return self.lang_embeddings_cache[cache_key]

        # Natural language conditioning: "Twi: egg", "Yoruba: harvest"
        conditioned_texts = [f"{lang_name}: {lbl}" for lbl in self.label_texts]
        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode(
                conditioned_texts,
                batch_size=256,
                convert_to_tensor=True,
                normalize_embeddings=True,
            ).to(self.device)
        self.lang_embeddings_cache[cache_key] = emb
        return emb

    def predict(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,  # Can be ISO code ("twi") or language name ("Twi")
        threshold: float = 0.35,
        top_k: int = 15,
        filter_type: Optional[str] = None,  # "noun_phrase" or "verb"
    ) -> PredictionResult:
        """Predict labels for an input audio file or waveform.

        Args:
            audio_input: File path (str/Path) to audio (.wav, .flac, .mp3).
            language: Optional ISO code (e.g. 'twi') or name (e.g. 'Twi') for disambiguation.
            threshold: Minimum cosine similarity score (between 0.0 and 1.0).
            top_k: Maximum number of matched labels to return.
            filter_type: Optionally restrict results to "noun_phrase" or "verb".

        Returns:
            PredictionResult: structured result containing matched labels and scores.
        """
        iso_code, lang_name = resolve_language(language)
        label_matrix = self.get_label_embeddings(lang_name=lang_name)

        audio_path_str = str(audio_input)
        try:
            info = sf.info(audio_path_str)
            duration = round(info.duration, 2)
        except Exception:
            duration = 0.0

        self.model.eval()
        with torch.no_grad():
            audio_emb = self.model.encode(
                [{"audio": audio_path_str}],
                batch_size=1,
                convert_to_tensor=True,
                normalize_embeddings=True,
            ).to(self.device)

            # Cosine similarity: [1, num_labels]
            sims = torch.mm(audio_emb, label_matrix.T).squeeze(0)
            scores = sims.float().cpu().numpy()

        ranked_indices = np.argsort(-scores)

        matched: List[LabelScore] = []
        for idx in ranked_indices:
            score_val = float(scores[idx])
            if score_val < threshold and len(matched) > 0:
                break
            l_type = self.label_types[idx]
            if filter_type and l_type != filter_type:
                continue

            matched.append(
                LabelScore(
                    label=self.label_texts[idx],
                    type=l_type,
                    score=round(score_val, 4),
                )
            )
            if len(matched) >= top_k:
                break

        top_pred = matched[0] if matched else None

        lang_info = {"iso": iso_code, "name": lang_name} if lang_name else None

        return PredictionResult(
            audio=audio_path_str,
            duration_seconds=duration,
            threshold=threshold,
            language_conditioned=lang_info,
            matched_count=len(matched),
            matched_labels=matched,
            top_prediction=top_pred,
        )
