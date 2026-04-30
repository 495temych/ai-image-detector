from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    run_id: str


class ExplainResponse(BaseModel):
    label: str
    confidence: float
    gradcam_base64: str
