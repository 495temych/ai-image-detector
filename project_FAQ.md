# AI Image Detector — Full Project Overview

The project is an **MLOps pipeline** — the goal isn't just a model that detects AI-generated images, but a complete system showing how a machine learning product gets built, versioned, evaluated, and deployed automatically.

---

## 1. Data Layer

**Tool: DVC + DagsHub**

The raw images live on DagsHub (a Git-compatible ML platform), not in the GitHub repo. Two DVC pointer files track what's stored remotely:

- `data.dvc` → points to the `data/` directory (dataset metadata + subset indices)
- `models.dvc` → points to the `models/` directory (checkpoints, weights, ONNX)

When someone runs `dvc pull`, DVC downloads the actual files from DagsHub. This keeps the Git repo lightweight while still pinning every run to an exact dataset/model snapshot.

The dataset is a ~500 images/class subset of a [Kaggle AI vs. Real dataset](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images), split 70/15/15 into train/val/test by `src/data/prepare_dataset.py`.

---

## 2. Training

**Tool: Kaggle GPU notebook + PyTorch**

Training happens **outside** the repo — on a Kaggle GPU notebook. EfficientNet-B0 (pretrained on ImageNet) is fine-tuned on the dataset. The resulting `.pt` weights are downloaded from Kaggle and versioned via DVC.

This was a deliberate design choice: Kaggle gives free GPU, zero local setup, and the weights are versioned anyway. CI never re-trains.

---

## 3. Evaluation

**Tool: MLflow + DagsHub + ONNX Runtime**

`src/evaluate.py` runs the ONNX model against the test split and computes:

- Accuracy
- F1 score
- AUC-ROC

Every run is logged to **MLflow** hosted on DagsHub with:
- The metrics above
- The git commit SHA (so you know exactly which code version)
- The DVC data hash (so you know exactly which dataset version)
- The ONNX model artifact

The run ID is saved to `mlflow_run_id.txt` and passed forward through the pipeline.

---

## 4. Model Registry

**Tool: MLflow Model Registry**

`src/register.py` reads the run's accuracy from MLflow and compares it against the threshold in `configs/eval_config.yaml` (currently 85%).

- **Below threshold** → the script exits with error code 1, the pipeline stops
- **Above threshold** → the model is registered in MLflow's model registry and promoted to **Production** stage

This is the quality gate between evaluation and deployment.

---

## 5. ONNX Export

**Tool: MLflow + PyTorch + ONNX**

`src/export_onnx.py` pulls the Production model from MLflow, converts it from PyTorch to ONNX format (framework-agnostic, faster inference), and uploads the `.onnx` file back to MLflow as an artifact.

The `api/model.onnx` committed to the repo is the current production ONNX file.

---

## 6. Serving — FastAPI

**Tool: FastAPI + ONNX Runtime + PyTorch (for GradCAM)**

`api/main.py` exposes three endpoints:

| Endpoint | What it does |
|---|---|
| `GET /health` | Returns model version + status |
| `POST /predict` | Runs the ONNX model → returns label + confidence |
| `POST /predict-explain` | Runs PyTorch model with GradCAM → returns label + confidence + heatmap image |

`api/model.py` loads both the ONNX model (for fast prediction) and the `.pt` weights (for GradCAM explainability) at startup. GradCAM highlights which regions of the image triggered the "real" or "fake" decision.

---

## 7. UI — Node.js Frontend

**Tool: Express + SQLite + vanilla JS**

`ui/server.js` is an Express server that:
- Proxies image uploads to the FastAPI `/predict` or `/predict-explain` endpoint
- Stores every prediction in a **SQLite database** (`ui/data/db.sqlite`)
- Serves three pages:

| Page | Purpose |
|---|---|
| `/` | Main detection UI — upload image, see result + GradCAM heatmap |
| `/benchmark` | Batch-test a set of images, see aggregate accuracy |
| `/dashboard` | Model health monitoring — session stats, confidence distributions, recent predictions |

The dashboard is the MLOps observability layer: you can see how the model is performing in production over time.

---

## 8. Docker

**Tool: Docker Compose**

`docker-compose.yml` runs the two services together:

```
api   →  port 8000  (FastAPI + ONNX model)
ui    →  port 3000  (Express frontend, proxies to api)
```

The UI container is built from the repo root so it can access `image_samples/`. The API container has the ONNX model and PyTorch weights baked in at build time. The SQLite DB is mounted as a volume so it persists across restarts.

---

## 9. CI/CD — GitHub Actions

Two workflows:

**`ci.yml`** — runs on every push and PR to `main`:

```
lint ──┐
       ├──► docker build + push to GHCR   (main only)
test ──┘
```

- `lint`: ruff checks `src/`, `api/`, `tests/`
- `test`: 13 unit tests with no model loading (heavy imports are lazy, mocked in tests)
- `docker`: builds both the API image and UI image and pushes them to GitHub Container Registry

**`ml-pipeline.yml`** — triggered manually when a new model is trained:

```
evaluate → register → export → docker
```

Each job depends on the previous one. If accuracy is below threshold, `register` fails and the chain stops — the old Production model stays in place.

---

## How It All Connects End-to-End

```
Kaggle (train)
    ↓ .pt weights
DagsHub/DVC (store & version)
    ↓ dvc pull
src/evaluate.py → MLflow run (metrics + model artifact)
    ↓ run_id
src/register.py → MLflow Model Registry (Production stage)
    ↓
src/export_onnx.py → model.onnx → api/model.onnx
    ↓
GitHub Actions → Docker build → GHCR image
    ↓
docker-compose up → api:8000 + ui:3000
    ↓
User uploads image → prediction + GradCAM heatmap
    ↓
SQLite → dashboard shows live model health
```

Every step is traceable: the Docker image tag is the git SHA, the MLflow run links back to the git commit and DVC data hash, and the dashboard shows production behaviour over time.
