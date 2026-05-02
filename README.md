# AI Image Authenticity Detector



Binary image classifier that flags whether an image is **AI-generated** or **real**.
Built as an end-to-end **MLOps pipeline** for the *Machine Learning and Data in
Operation* (TSM_MachLeData) course at ZHAW.

---

## Project Goal

The deliverable of this project is the **pipeline**, not the model.
Given an image → output `real` or `ai-generated` with a confidence score —
produced by a workflow that is reproducible, tracked, automated, and deployable.

**In scope**
- Data versioning tied to every training run
- Experiment tracking (metrics, params, artifacts) for every run
- A model registry with clear promotion stages
- Continuous integration that retrains, evaluates, and registers on every push
- Packaged serving (REST + demo UI) behind a portable container

**Not in scope**
- Beating state-of-the-art accuracy on AI-vs-real detection
- Building a novel model architecture — we fine-tune an off-the-shelf backbone

---

## Why This Project

Generative image models have crossed the threshold of photorealism, which makes
provenance-and-authenticity tooling increasingly useful (journalism, moderation,
identity verification). A binary `real` / `ai-generated` classifier is a
well-scoped, well-documented problem — which lets us spend our time on the
*operations* layer instead of on task-specific research.

---

## Pipeline Overview

```
 Kaggle notebook (EfficientNet fine-tuning)   ← training happens here, externally
      │
      │  download trained .pt weights
      ▼
  DVC  ───────────────┐  (data + model versioning; each run pinned to a data hash)
      │               │
      ▼               │
  Evaluate on         │
  test split   ───────┼──▶ MLflow  (metrics, params, artifacts, model registry)
      │               │
      ▼               │
  ONNX export ────────┘
      │
      ▼
  GitHub Actions (CI: evaluate → register → export → docker, on push to main)
      │
      ▼
  FastAPI + Docker (REST endpoint serving the ONNX model)
      │
      ▼
  Streamlit (demo UI)
```

Training runs once externally on Kaggle ([notebook](https://www.kaggle.com/code/marcosncosta/ai-vs-real-image-efficientnet-fine-tuning-86380e)).
Every registered model is traceable back to **(git commit, DVC data version,
MLflow run)** — that triple is what makes the pipeline reproducible.

---

## What Gets Tracked

| Artifact | Where it lives | Linked to |
|----------|----------------|-----------|
| Source code | Git | commit SHA |
| Raw + processed images | DVC (external remote) | `.dvc` file committed to Git |
| Trained EfficientNet weights (`.pt`) | DVC + MLflow artifact store | MLflow run ID |
| Hyperparameters, metrics, plots | MLflow tracking server | MLflow run ID |
| Exported `.onnx` model | MLflow artifact store | MLflow run ID |
| Promoted models (Staging / Production) | MLflow model registry | model version |
| CI runs (evaluate → register → export → docker) | GitHub Actions | workflow run |

---

## Repo Structure

```
ai-image-detector/
├── data/
│   ├── raw/             # original Kaggle images (DVC-tracked, not in Git)
│   └── processed/       # train/val/test splits (DVC-tracked)
├── src/
│   ├── data/            # dataset loading and preprocessing
│   ├── models/          # model definition (EfficientNet-B0 backbone)
│   ├── train.py         # training entrypoint (logs to MLflow)
│   ├── evaluate.py      # evaluation script
│   ├── export_onnx.py   # PyTorch → ONNX export
│   └── predict.py       # single-image inference
├── api/
│   ├── main.py          # FastAPI app (serves ONNX model)
│   └── Dockerfile
├── app/
│   └── app.py           # Streamlit demo UI
├── configs/
│   └── train_config.yaml  # all hyperparameters in one place
├── notebooks/           # exploratory analysis
├── .github/workflows/   # GitHub Actions CI pipeline
├── pitch_decks/         # course pitch-deck sources
├── examples/            # reference decks & presentations
├── env.yaml             # conda environment
└── README.md
```

---

## Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-org>/ai-image-detector.git
cd ai-image-detector
conda env create -f env.yaml
conda activate mlops-img
```

### 2. Pull the dataset via DVC

```bash
dvc pull
```

> First-time setup — configure the DVC remote (see `docs/dvc_setup.md`).

### 3. Start the MLflow tracking server (local)

```bash
mlflow server --host 127.0.0.1 --port 5000
```

### 4. Download trained model weights

The EfficientNet model is trained externally on Kaggle
([notebook](https://www.kaggle.com/code/marcosncosta/ai-vs-real-image-efficientnet-fine-tuning-86380e)).
Download the resulting `.pt` file and place it at the path specified in
`configs/eval_config.yaml` (default: `model/efficientnet.pt`).

Then run evaluation (logs metrics and the weights to MLflow):

```bash
python src/evaluate.py --config configs/eval_config.yaml
```

All metrics, params, and artifacts are logged to MLflow. Open the UI at
`http://127.0.0.1:5000`.

### 5. Export to ONNX

```bash
python src/export_onnx.py --run-id <mlflow_run_id>
```

### 6. Run the API locally

```bash
cd api
uvicorn main:app --reload
```

### 7. Run the Streamlit demo

```bash
streamlit run app/app.py
```

---

## Tools (by stage)

| Stage | Tool | Role |
|-------|------|------|
| Data & model versioning | DVC + Git | Pin dataset and model weights to a commit |
| Training | PyTorch + Kaggle | Fine-tune EfficientNet on Kaggle GPU (external to CI) |
| Experiment tracking | MLflow | Metrics, params, artifacts, model registry |
| Portable inference | ONNX | Framework-agnostic exported model |
| Automation / CI | GitHub Actions | Evaluate → register → export → docker on push to `main` |
| Serving | FastAPI + Docker | Containerized REST endpoint |
| Demo UI | Streamlit | Interactive image upload + prediction |

---

## Dataset

[AI Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)
Kaggle · `tristanzhang32` · Binary: `real` / `fake`
Used for academic purposes only.

---

## Team

<!-- Fill in once finalized -->
- [Name 1]
- [Name 2]
- [Name 3]
- [Name 4]

ZHAW School of Engineering · Machine Learning and Data in Operation (Spring 2026)
