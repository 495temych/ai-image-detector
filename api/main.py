from fastapi import FastAPI, File, HTTPException, UploadFile

from api.schemas import ExplainResponse, HealthResponse, PredictResponse
from api.model import MODEL_VERSION, RUN_ID, explain_gradcam, predict_onnx

app = FastAPI(title="AI Image Detector", version="1.0.0")

_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")
    return file.file.read()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=MODEL_VERSION)


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)) -> PredictResponse:
    image_bytes = _read_image(file)
    label, confidence = predict_onnx(image_bytes)
    return PredictResponse(
        label=label,
        confidence=confidence,
        model_version=MODEL_VERSION,
        run_id=RUN_ID,
    )


@app.post("/predict-explain", response_model=ExplainResponse)
def predict_explain(file: UploadFile = File(...)) -> ExplainResponse:
    image_bytes = _read_image(file)
    label, confidence, gradcam_b64 = explain_gradcam(image_bytes)
    return ExplainResponse(label=label, confidence=confidence, gradcam_base64=gradcam_b64)
