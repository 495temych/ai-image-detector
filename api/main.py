import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "model.onnx")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.json")
FEATURE_EXTRACTOR_NAME = "dima806/ai_vs_real_image_detection"

app = FastAPI(title="AI Image Detector", version="1.0")

# Lazy-loaded singletons — initialised on first request, not at import time.
# This makes the module importable in tests without real model files.
_session = None
_feature_extractor = None
_id2label = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(MODEL_PATH)
    return _session


def _get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        from transformers import AutoFeatureExtractor
        _feature_extractor = AutoFeatureExtractor.from_pretrained(
            FEATURE_EXTRACTOR_NAME
        )
    return _feature_extractor


def _get_id2label() -> dict[str, str]:
    global _id2label
    if _id2label is None:
        _id2label = json.loads(Path(LABELS_PATH).read_text())
    return _id2label


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _predict(image: Image.Image) -> dict:
    session = _get_session()
    feature_extractor = _get_feature_extractor()
    id2label = _get_id2label()

    inputs = feature_extractor(images=image.convert("RGB"), return_tensors="np")
    pixel_values = inputs["pixel_values"].astype(np.float32)
    logits = session.run(None, {"pixel_values": pixel_values})[0][0]
    probs = _softmax(logits)

    pred_idx = int(np.argmax(probs))
    raw_label = id2label[str(pred_idx)].upper()
    label = "real" if raw_label == "REAL" else "ai-generated"
    confidence = float(probs[pred_idx])
    return {"label": label, "confidence": round(confidence, 4)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = _predict(image)
    return JSONResponse(result)
