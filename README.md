# AI Image Authenticity Detector

Binary image classifier that detects whether an image is **AI-generated** or **real**.  
Built as an end-to-end MLOps pipeline for the *Machine Learning and Data in Operation* course at ZHAW.

**Model:** EfficientNet-B0 · Test acc: 89.6% · AUC: 96.2%

---

## Web UI

![UI preview — compare mode showing original vs GradCAM heatmap side by side](imgs/UI-explain-preview.png)

A Node.js frontend that lets you drag-and-drop any image and instantly see whether the model considers it real or AI-generated, along with a GradCAM explanation of which regions drove the decision.

**Key features:**

- **Drag & drop / paste / browse** image upload with live preview
- **Verdict card** — Real or AI-generated with animated confidence gauge
- **Three GradCAM view modes:**
  - *Original* — the uploaded image, no overlay
  - *Blend* — GradCAM heatmap faded over the image via an opacity slider (0–100%)
  - *Compare* — draggable vertical divider splits the frame: original on the left, GradCAM on the right
- **JET colormap legend** — blue = low activation, red = high activation
- **Model attention thumbnail** in the result card shows where the model focused
- **Feedback system** — mark predictions correct or wrong; saves to `ui/feedback/feedback.jsonl` for future retraining

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
  FastAPI (REST endpoints: /predict, /predict-explain)
     │
     ▼
  Node.js + Express (web UI on port 3000)
```

---

## Repo Structure

```
ai-image-detector/
├── api/
│   ├── main.py               # FastAPI app — /health, /predict, /predict-explain
│   ├── model.py              # ONNX inference + PyTorch GradCAM
│   ├── schemas.py            # Pydantic response models
│   ├── download_artifacts.py # pull model files from DagsHub S3
│   ├── export_onnx.py        # re-export ONNX from .pt weights
│   ├── requirements.txt
│   └── .env.example          # secrets template
├── ui/
│   ├── server.js             # Express server — proxies /api/* → FastAPI, handles feedback
│   ├── package.json
│   ├── public/
│   │   ├── index.html        # single-page app
│   │   ├── script.js         # upload, classify, GradCAM view modes, feedback
│   │   └── styles.css        # design system
│   └── feedback/
│       ├── feedback.jsonl    # one JSON entry per user submission
│       └── images/           # uploaded images saved with UUID filenames
├── picture_samples/
│   ├── fake/                 # 10 AI-generated test images
│   └── real/                 # 10 real test images
├── imgs/                     # screenshots and assets
├── data/
│   └── dataset.yaml
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

### 4. Start the FastAPI backend

```bash
uvicorn api.main:app --port 8000
```

Swagger UI → http://localhost:8000/docs

### 5. Start the web UI

In a second terminal:

```bash
cd ui
npm install   # first time only
npm start
```

Open **http://localhost:3000**

The UI health pill in the top-right corner turns green once the API is reachable.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Status check |
| `POST` | `/predict` | Classify image → label + confidence (ONNX, fast) |
| `POST` | `/predict-explain` | Classify + GradCAM heatmap (PyTorch, ~1 s) |

### Example requests

```bash
# health check
curl http://localhost:8000/health

# predict
curl -X POST http://localhost:8000/predict \
  -F "file=@picture_samples/fake/0001.jpg"

# predict + GradCAM explanation
curl -X POST http://localhost:8000/predict-explain \
  -F "file=@picture_samples/real/0001.jpg"
```

### Response shapes

```json
// GET /health
{ "status": "ok", "model": "efficientnet_v1" }

// POST /predict
{
  "label": "fake",
  "confidence": 0.98,
  "model_version": "efficientnet_v1",
  "run_id": "2561d86a..."
}

// POST /predict-explain
{
  "label": "real",
  "confidence": 0.91,
  "gradcam_base64": "<base64-encoded PNG>"
}
```

Render the GradCAM image in a browser:
```html
<img src="data:image/png;base64,{{ gradcam_base64 }}" />
```

The GradCAM PNG is a 224×224 blend of the original image (50%) and a JET-colormap heatmap (50%). Red regions indicate the highest model activation; blue regions the lowest.

---

## Tools

| Tool | Role |
|------|------|
| PyTorch | EfficientNet-B0 fine-tuning + GradCAM |
| ONNX Runtime | Fast inference in `/predict` |
| FastAPI | REST API |
| Node.js + Express | Web UI server |
| DVC | Data versioning |
| MLflow | Experiment tracking + model registry |
| GitHub Actions | CI pipeline |
| DagsHub | Remote storage + MLflow tracking server |

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

- Marcos
- Artemi
- Afshin
- Jibin

ZHAW School of Engineering · Machine Learning and Data in Operation
