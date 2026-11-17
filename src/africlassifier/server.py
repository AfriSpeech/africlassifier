"""FastAPI REST serving microservice for africlassifier (CPU-First by default)."""

import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from africlassifier.classifier import AfriClassifier, PredictionResult


class TextQueryRequest(BaseModel):
    text: str
    language: Optional[str] = None
    target_language: str = "en"  # "en" (default) or "fr"
    threshold: float = 0.15
    top_k: int = 5


def create_app(
    model_path: str = "outputs/embeddinggemma_lora",
    indices_dir: str = "data/indices",
    device: str = "cpu",
    quantize_cpu: bool = False,
) -> FastAPI:
    app = FastAPI(
        title="africlassifier Serving API",
        description="CPU-First Multimodal Classification for African Speech & Text into English/French AfricaNouns.",
        version="0.3.0",
    )

    classifier = AfriClassifier.load(
        model_path=model_path,
        indices_dir=indices_dir,
        device=device,
        quantize_cpu=quantize_cpu,
    )

    @app.post("/classify/text")
    def classify_text(req: TextQueryRequest):
        try:
            res = classifier.predict(
                input_data=req.text,
                language=req.language,
                target_language=req.target_language,
                threshold=req.threshold,
                top_k=req.top_k,
            )
            return res.to_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/classify/audio")
    def classify_audio(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
        target_language: str = Form("en"),
        threshold: float = Form(0.15),
        top_k: int = Form(5),
    ):
        suffix = Path(file.filename or "audio.flac").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        try:
            res = classifier.predict(
                input_data=tmp_path,
                language=language,
                target_language=target_language,
                threshold=threshold,
                top_k=top_k,
            )
            return res.to_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return app


if __name__ == "__main__":
    import uvicorn
    device_choice = "cuda" if torch.cuda.is_available() and os.environ.get("USE_GPU") == "1" else "cpu"
    app = create_app(device=device_choice)
    uvicorn.run(app, host="0.0.0.0", port=8000)
