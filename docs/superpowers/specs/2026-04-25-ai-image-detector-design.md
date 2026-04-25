# AI Image Detector — System Design Spec

**Date:** 2026-04-25
**Course:** TSM_MachLeData · ZHAW School of Engineering · Spring 2026
**Team:** Marcos, Artemi, Afshin, Jibin

---

## 1. Goal

Build a complete, reproducible **MLOps pipeline** that classifies an image as `real` or `ai-generated`. The deliverable is the pipeline, not model accuracy. Every registered model must be traceable to a triple: `(git commit, DVC data hash, MLflow run ID)`.

---

## 2. Architecture Overview

```
[DATA LAYER]
  Kaggle dataset (subset ~500/class) → train/val/test splits
  → tracked with DVC → pushed to DagsHub remote

[MODEL LAYER]
  HuggingFace model pulled → evaluated on DVC test split
  → metrics + model logged to MLflow (DagsHub)
  → if metrics pass threshold → promoted to Production in MLflow registry
  → exported to ONNX

[CI LAYER]  (triggered on push to main)
  GitHub Actions:
    job 1: evaluate  → log to MLflow
    job 2: register  → promote if threshold met
    job 3: export    → ONNX artifact
    job 4: docker    → build image → push to GHCR

[SERVING LAYER]
  Docker container (FastAPI + ONNX) running locally
  ← Streamlit app calls POST /predict
```

---

## 3. Tools

| Stage | Tool | Role |
|---|---|---|
| Source control | Git + GitHub | Every model links back to its commit SHA |
| Data versioning | DVC | Pins each dataset snapshot to a commit hash |
| DVC + MLflow remote | DagsHub | Hosts DVC remote storage and MLflow tracking server (free tier) |
| Model source | HuggingFace (`dima806/ai_vs_real_image_detection`) | Pre-trained ViT-base, Apache 2.0 |
| Experiment tracking | MLflow | Metrics, params, artifacts, model registry |
| Portable inference | ONNX | Framework-agnostic exported model |
| CI/CD | GitHub Actions | Eval → register → export → Docker build → GHCR push |
| Serving | FastAPI + Docker | Containerised REST endpoint (`POST /predict`) |
| Demo UI | Streamlit | Calls FastAPI `/predict`, displays label + confidence |

Minimum tool count: **7 distinct tools** (DVC, MLflow, HuggingFace, GitHub Actions, ONNX, FastAPI/Docker, Streamlit). Meets the ≥3 requirement with significant margin.

---

## 4. Model

- **Source:** `dima806/ai_vs_real_image_detection` on HuggingFace
- **Architecture:** ViT-base-patch16-224 fine-tuned for binary classification
- **License:** Apache 2.0
- **Baseline metrics (reported):** accuracy 98.25%, F1 0.9826
- **Why not training from scratch:** The dataset is 52GB; training an EfficientNet from scratch would be computationally expensive and is not the project goal. Using a pre-trained model and focusing CI on evaluation, registration, and deployment is a realistic MLOps pattern.
- **Note on concept drift:** The model was trained on an older dataset. This is an intentional talking point — the pipeline is designed so that when a better model becomes available, swapping it requires only a config change, not a pipeline rewrite.

---

## 5. Data Layer

### Dataset

- **Source:** [AI Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images) — Kaggle · `tristanzhang32`
- **Full size:** 52.41 GB (not downloaded in full)
- **Working subset:** ~500 images per class (real / fake) → ~100 MB total
- **Why a subset:** Evaluation of a pre-trained model on 500 images per class produces statistically meaningful metrics. DVC versioning works identically on 100 MB as on 52 GB. DagsHub free tier is 10 GB.

### Setup (one-time, done by one team member)

```bash
# 1. Download and unzip via Kaggle API
kaggle datasets download tristanzhang32/ai-generated-images-vs-real-images \
  -p data/raw --unzip

# 2. Run split script — keeps 500/class, deletes the rest, creates splits
python src/data/prepare_dataset.py --per-class 500 --split 0.7/0.15/0.15

# 3. Version with DVC and push to DagsHub
dvc add data/raw data/processed
git add data/raw.dvc data/processed.dvc .dvc/config
git commit -m "add dataset splits (500/class)"
dvc push
```

### DVC remote config (`.dvc/config`)

```ini
[core]
    remote = dagshub
[remote "dagshub"]
    url = https://dagshub.com/<username>/ai-image-detector.dvc
```

### Git vs DVC tracking

| File | Where |
|---|---|
| `data/raw/` (images) | DVC only — never Git |
| `data/processed/` (splits) | DVC only |
| `data/raw/.gitkeep` | Git (empty placeholder) |
| `.dvc/config` | Git |
| `data/raw.dvc`, `data/processed.dvc` | Git (hash pointers) |

### Reproducing on a new machine

```bash
dvc pull   # fetches the exact versioned subset from DagsHub
```

---

## 6. CI/CD Pipeline (GitHub Actions)

**File:** `.github/workflows/ci.yml`
**Trigger:** push to `main`

### Jobs

```
evaluate → register → export → docker
```

#### Job 1: evaluate
- `dvc pull` — fetch pinned dataset
- Load model from HuggingFace
- Run inference on `data/processed/test/`
- Log to MLflow: accuracy, F1, AUC-ROC, DVC data hash, git commit SHA
- MLflow tracking URI: `https://dagshub.com/<username>/ai-image-detector.mlflow`

#### Job 2: register
- Read accuracy from the completed MLflow run
- If accuracy ≥ threshold (default `0.90`, set in `configs/eval_config.yaml`):
  - Register model version in MLflow registry
  - Promote to `Production`
- If below threshold: fail the workflow with a clear message

#### Job 3: export
- Download `Production` model weights from MLflow registry
- Export to ONNX (`model.onnx`)
- Upload ONNX file as MLflow artifact on the same run

#### Job 4: docker
- `docker build` the API image from `api/Dockerfile`
- `docker push` to GitHub Container Registry (`ghcr.io/<org>/ai-image-detector-api:latest`)
- Tags image with git commit SHA for traceability

### GitHub Secrets required

| Secret | Value |
|---|---|
| `DAGSHUB_TOKEN` | DagsHub personal access token |
| `DAGSHUB_USERNAME` | DagsHub username |
| `KAGGLE_USERNAME` | Kaggle username (for `dvc pull` if needed) |
| `KAGGLE_KEY` | Kaggle API key |

---

## 7. Serving Layer

### FastAPI (`api/main.py`)

- Single endpoint: `POST /predict`
- Accepts: multipart image file upload
- Loads `model.onnx` at startup
- Preprocesses image (resize to 224×224, normalise)
- Returns: `{"label": "real" | "ai-generated", "confidence": 0.97}`

### Dockerfile (`api/Dockerfile`)

- Base: `python:3.11-slim`
- Installs: `fastapi`, `uvicorn`, `onnxruntime`, `Pillow`
- Copies `model.onnx` into the image at build time
- Exposes port `8000`

### Running locally for the demo

```bash
docker pull ghcr.io/<org>/ai-image-detector-api:latest
docker run -p 8000:8000 ghcr.io/<org>/ai-image-detector-api:latest
```

---

## 8. Demo UI (Streamlit)

**File:** `app/app.py`

- File uploader (jpg/png)
- On upload: sends image to `http://localhost:8000/predict`
- Displays: prediction label (`REAL` / `AI-GENERATED`) + confidence percentage bar
- Pre-load 6–8 test images for the live demo: mix of obvious real, obvious AI-generated, and 2–3 ambiguous cases

### Running

```bash
streamlit run app/app.py
```

Requires the FastAPI Docker container to be running on port 8000.

---

## 9. DagsHub Setup (one-time)

1. Go to [dagshub.com](https://dagshub.com) → New Project → **Connect a repository** → link `ai-image-detector` GitHub repo
2. DagsHub generates:
   - DVC remote URL: `https://dagshub.com/<username>/ai-image-detector.dvc`
   - MLflow tracking URL: `https://dagshub.com/<username>/ai-image-detector.mlflow`
3. Generate a DagsHub token → save as `DAGSHUB_TOKEN` in GitHub Actions secrets and as a local env var
4. Set DVC remote in `.dvc/config` (see Section 5)
5. Set MLflow tracking URI in scripts and CI: `MLFLOW_TRACKING_URI=https://dagshub.com/<username>/ai-image-detector.mlflow`

### Migration to GCS (when credits arrive)

The migration requires only two changes — no code changes:

1. **DVC remote:** Update `.dvc/config` remote URL from DagsHub to `gs://<bucket>/dvc`, update GitHub secret with GCS credentials, run `dvc push` to the new remote
2. **MLflow:** Update `MLFLOW_TRACKING_URI` env var to point to a GCS-backed or self-hosted MLflow instance

Everything else (scripts, CI jobs, Docker, Streamlit) remains unchanged.

---

## 10. Repository Structure

```
ai-image-detector/
├── data/
│   ├── raw/             # Kaggle subset (DVC-tracked, not in Git)
│   └── processed/       # train/val/test splits (DVC-tracked)
├── src/
│   ├── data/
│   │   └── prepare_dataset.py   # download subset + create splits
│   ├── evaluate.py              # eval on test split → log to MLflow
│   ├── register.py              # promote model if threshold met
│   └── export_onnx.py           # download Production model → ONNX
├── api/
│   ├── main.py                  # FastAPI app (POST /predict)
│   └── Dockerfile
├── app/
│   └── app.py                   # Streamlit demo UI
├── configs/
│   └── eval_config.yaml         # eval threshold, model ID, paths
├── .github/
│   └── workflows/
│       └── ci.yml               # evaluate → register → export → docker
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-25-ai-image-detector-design.md  # this file
├── site/                        # existing HTML project page (unchanged)
├── pitch_decks/                 # existing pitch deck (unchanged)
├── imgs/                        # tool logos for site (unchanged)
├── env.yaml                     # conda environment
├── .dvc/
│   └── config                   # DVC remote config (DagsHub)
├── .gitignore
└── README.md
```

---

## 11. Presentation Demo Plan

The live demo during the final presentation uses only the Streamlit app and the running Docker container. All other pipeline components (MLflow registry, GitHub Actions run history, DVC data versions) are shown via screenshots on slides.

**Demo run order:**
1. Show Docker container running (`docker ps`)
2. Open Streamlit at `localhost:8501`
3. Upload 6–8 pre-selected images in sequence:
   - 2 clearly real photographs → expect `REAL` with high confidence
   - 2 clearly AI-generated images → expect `AI-GENERATED` with high confidence
   - 2–3 ambiguous images → discuss confidence scores
4. Point out the confidence score bar on each result

**Fallback:** If the Docker container fails to start, Streamlit can call the model directly (load ONNX locally) — add a `USE_LOCAL_MODEL` env flag to `app.py` as a fallback path.

---

## 12. Out of Scope

- Training a model from scratch
- State-of-the-art detection accuracy
- Novel model architecture
- Adversarial robustness
- Cloud deployment (everything runs locally for the demo)
