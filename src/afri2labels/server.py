"""High-performance FastAPI serving engine for afri2labels."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from unsloth import FastSentenceTransformer

from afri2labels.languages import resolve_language


class MatchedLabel(BaseModel):
    rank: int
    label: str
    type: str
    score: float


class PredictionResponse(BaseModel):
    input_type: str  # "audio" or "text"
    language: Optional[dict]
    threshold: float
    matched_count: int
    matched_labels: List[MatchedLabel]
    top_prediction: Optional[MatchedLabel]
    inference_time_ms: float


class TextQueryRequest(BaseModel):
    text: str
    language: Optional[str] = None
    threshold: float = 0.30
    top_k: int = 10


def create_app(
    model_path: str = "outputs/checkpoint-1000",
    index_path: str = "data/deployment_index/label_embeddings.pt",
    metadata_path: str = "data/deployment_index/label_metadata.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> FastAPI:
    app = FastAPI(
        title="afri2labels Serving API",
        description="Fast multimodal retrieval of top-K concepts and named entities from African speech and text.",
        version="0.2.0",
    )

    print(f"Loading afri2labels model from {model_path} on {device}...")
    model = FastSentenceTransformer.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        config_kwargs={"vision_config": None},
        full_finetuning=False,
    )
    model.eval()

    print(f"Loading vector index from {index_path}...")
    label_embeddings = torch.load(index_path, map_location=device).float()  # [N, 768]
    # Ensure L2 normalization
    label_embeddings = torch.nn.functional.normalize(label_embeddings, p=2, dim=-1)

    print(f"Loading label metadata from {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    labels_list = [item["label"] for item in metadata]
    types_list = [item.get("type", "concept") for item in metadata]

    print(f"afri2labels API ready! Loaded {len(labels_list):,} concepts.")

    @app.post("/predict/text", response_model=PredictionResponse)
    def predict_text(req: TextQueryRequest):
        t0 = time.time()
        iso_code, lang_name = resolve_language(req.language)
        prefix = f"{lang_name}: " if lang_name else ""
        prompt = f"{prefix}{req.text.strip()}"

        with torch.no_grad():
            q_emb = model.encode(
                [prompt],
                batch_size=1,
                convert_to_tensor=True,
                normalize_embeddings=True,
            ).to(device)

            # Dot-product Top-K search
            sims = torch.mm(q_emb, label_embeddings.T).squeeze(0)  # [N]
            scores, indices = torch.topk(sims, k=min(req.top_k, len(labels_list)))

        scores = scores.float().cpu().numpy()
        indices = indices.cpu().numpy()

        matched = []
        for rank, (idx, score_val) in enumerate(zip(indices, scores), 1):
            if score_val < req.threshold and len(matched) > 0:
                break
            matched.append(
                MatchedLabel(
                    rank=rank,
                    label=labels_list[idx],
                    type=types_list[idx],
                    score=round(float(score_val), 4),
                )
            )

        top_pred = matched[0] if matched else None
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        return PredictionResponse(
            input_type="text",
            language={"iso": iso_code, "name": lang_name} if lang_name else None,
            threshold=req.threshold,
            matched_count=len(matched),
            matched_labels=matched,
            top_prediction=top_pred,
            inference_time_ms=elapsed_ms,
        )

    @app.post("/predict/audio", response_model=PredictionResponse)
    def predict_audio(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
        threshold: float = Form(0.30),
        top_k: int = Form(10),
    ):
        t0 = time.time()
        iso_code, lang_name = resolve_language(language)

        suffix = Path(file.filename or "audio.flac").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        try:
            with torch.no_grad():
                q_emb = model.encode(
                    [{"audio": tmp_path}],
                    batch_size=1,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                ).to(device)

                sims = torch.mm(q_emb, label_embeddings.T).squeeze(0)
                scores, indices = torch.topk(sims, k=min(top_k, len(labels_list)))

            scores = scores.float().cpu().numpy()
            indices = indices.cpu().numpy()

            matched = []
            for rank, (idx, score_val) in enumerate(zip(indices, scores), 1):
                if score_val < threshold and len(matched) > 0:
                    break
                matched.append(
                    MatchedLabel(
                        rank=rank,
                        label=labels_list[idx],
                        type=types_list[idx],
                        score=round(float(score_val), 4),
                    )
                )

            top_pred = matched[0] if matched else None
            elapsed_ms = round((time.time() - t0) * 1000, 2)

            return PredictionResponse(
                input_type="audio",
                language={"iso": iso_code, "name": lang_name} if lang_name else None,
                threshold=threshold,
                matched_count=len(matched),
                matched_labels=matched,
                top_prediction=top_pred,
                inference_time_ms=elapsed_ms,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
