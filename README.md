# AI Image Authenticity Detector

Binary image classifier that detects whether an image is **AI-generated** or **real**.  
Built as an end-to-end MLOps pipeline for the *Machine Learning and Data in Operation* course at ZHAW.

**Model:** EfficientNet-B0 · Test acc: 0.8963 · AUC: 0.9616

---

## Pipeline Overview

```
Kaggle dataset
     │
     ▼
  DVC (data versioning)
     │
     ▼
  PyTorch training  ──→  MLflow (experiment tracking + model registry)
     │
     ▼
  GitHub Actions (CI: train → evaluate → register on every push)
     │
     ▼
  FastAPI (REST endpoints)
     │
     ▼
  Streamlit (demo UI)
```

---

## Repo Structure

```
ai-image-detector/
├── api/
│   ├── main.py               # FastAPI app — 3 endpoints
│   ├── model.py              # ONNX inference + GradCAM
│   ├── schemas.py            # Pydantic response models
│   ├── download_artifacts.py # pull model files from S3
│   ├── export_onnx.py        # re-export ONNX from .pt weights
│   ├── requirements.txt
│   └── .env.example          # secrets template
├── picture_samples/
│   ├── fake/                 # 10 AI-generated test images
│   └── real/                 # 10 real test images
├── data/
│   └── dataset.yaml
├── src/                      # training code (coming)
├── ui/                       # Streamlit UI (coming)
├── env.yaml                  # conda environment
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://dagshub.com/495temych/ai-image-detector.git
cd ai-image-detector
conda env create -f env.yaml
conda activate mlops-img
```

### 2. Configure secrets

```bash
cp api/.env.example api/.env
# fill in DAGSHUB_KEY_ID and DAGSHUB_S3 with your DagsHub credentials
```

### 3. Pull model artifacts from S3

```bash
python -m api.download_artifacts
```

This downloads `api/model.onnx` and `api/best_weights.pt` (~16 MB each).

> **Alternative:** if you already have `best_weights.pt`, re-export ONNX locally:
> ```bash
> python -m api.export_onnx
> ```

### 4. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger UI → http://localhost:8000/docs

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Status check |
| `POST` | `/predict` | Classify image → label + confidence |
| `POST` | `/explain` | Classify + GradCAM heatmap |

### Example requests

```bash
# health check
curl http://localhost:8000/health

# predict
curl -X POST http://localhost:8000/predict \
  -F "file=@picture_samples/fake/0001.jpg"

# explain (returns gradcam_base64 PNG)
curl -X POST http://localhost:8000/explain \
  -F "file=@picture_samples/real/0001.jpg"
```

### Response shapes

```json
// GET /health
{ "status": "ok", "model": "efficientnet_v1" }

// POST /predict
{ "label": "fake", "confidence": 0.98, "model_version": "efficientnet_v1", "run_id": "2561d86a..." }

// POST /explain
{ "label": "real", "confidence": 0.91, "gradcam_base64": "<base64 PNG>" }
```

Render the GradCAM image in a browser:
```html
<img src="data:image/png;base64,{{ gradcam_base64 }}" />
```

---

## Tools

| Tool | Role |
|------|------|
| DVC | Data versioning |
| MLflow | Experiment tracking + model registry |
| GitHub Actions | CI pipeline |
| PyTorch | EfficientNet-B0 fine-tuning |
| ONNX Runtime | Fast inference in `/predict` |
| FastAPI | REST API |
| Streamlit | Demo UI |

---

## Dataset

[AI Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)  
Kaggle · `tristanzhang32` · Binary: `real` / `fake` · Used for academic purposes only.

---

## MLflow

Tracking server: https://dagshub.com/495temych/ai-image-detector.mlflow  
Champion model: `efficientnet-b0 @champion` · Run ID: `2561d86a3f22495e91b2cc7d3d1d3497`

---

## Team

<!-- Add your names here -->
- [Name 1]
- [Name 2]
- [Name 3]
- [Name 4]

ZHAW School of Engineering · Machine Learning and Data in Operation
