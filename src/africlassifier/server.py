"""FastAPI REST microservice for africlassifier multi-task classification."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from africlassifier.classifier import AfriClassifier


class TextClassificationRequest(BaseModel):
    text: str
    language: Optional[str] = None
    top_k: int = 1


def create_app(
    model_path: str = "outputs/embeddinggemma_lora",
    taxonomy_index_path: str = "data/taxonomies_index.pt",
    device: str = "cpu",
) -> FastAPI:
    app = FastAPI(
        title="africlassifier Multi-Task Serving API",
        description="CPU-First Multi-Task Classifier for African Speech & Text (Language, Domain, Sentiment, Emotion, Intent).",
        version="0.3.0",
    )

    classifier = AfriClassifier.load(
        model_path=model_path,
        taxonomy_index_path=taxonomy_index_path,
        device=device,
    )

    @app.post("/classify/text")
    def classify_text(req: TextClassificationRequest):
        try:
            res = classifier.predict(
                input_data=req.text,
                language=req.language,
                top_k=req.top_k,
            )
            return res.to_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/classify/audio")
    def classify_audio(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
        top_k: int = Form(1),
    ):
        suffix = Path(file.filename or "audio.flac").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        try:
            res = classifier.predict(
                input_data=tmp_path,
                language=language,
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
