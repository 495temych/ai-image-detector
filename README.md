# AI Image Authenticity Detector

Binary image classifier that detects whether an image is **AI-generated** or **real**.  
Built as an end-to-end MLOps pipeline for the *Machine Learning and Data in Operation* course at ZHAW.

---

## Project Goal

Train and deploy a reproducible ML pipeline — not just a model.  
Given an image → output `real` or `ai-generated` with a confidence score.

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
  FastAPI + Docker (REST endpoint)
     │
     ▼
  Streamlit (demo UI)
```

---

## Repo Structure

```
ai-image-detector/
├── data/
│   ├── raw/            # original Kaggle images (DVC-tracked, not in Git)
│   └── processed/      # train/val/test splits (DVC-tracked)
├── src/
│   ├── data/           # dataset loading and preprocessing
│   ├── models/         # model definition (EfficientNet-B0 backbone)
│   ├── train.py        # training entrypoint
│   ├── evaluate.py     # evaluation script
│   └── predict.py      # single-image inference
├── api/
│   ├── main.py         # FastAPI app
│   └── Dockerfile
├── app/
│   └── app.py          # Streamlit demo
├── configs/
│   └── train_config.yaml  # all hyperparameters in one place
├── notebooks/          # exploratory analysis
├── .github/
│   └── workflows/      # GitHub Actions CI pipeline
├── env.yaml            # conda environment
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/<your-org>/ai-image-detector.git
cd ai-image-detector
conda env create -f env.yaml
conda activate mlops-img
```

### 2. Pull dataset via DVC

```bash
dvc pull
```

> First time: configure your DVC remote (see `docs/dvc_setup.md`).

### 3. Train

```bash
python src/train.py --config configs/train_config.yaml
```

All metrics and artifacts are logged to MLflow. Launch the UI with:

```bash
mlflow ui
```

### 4. Run the API locally

```bash
cd api
uvicorn main:app --reload
```

### 5. Run the Streamlit demo

```bash
streamlit run app/app.py
```

---

## Tools

| Tool | Role |
|------|------|
| DVC | Data versioning — tracks image dataset versions linked to each model run |
| MLflow | Experiment tracking + model registry — logs metrics, promotes best model to Production |
| GitHub Actions | CI — auto-triggers train → evaluate → register on push to `main` |
| PyTorch | Model training — EfficientNet-B0 fine-tuned for binary classification |
| FastAPI + Docker | REST API — serves the ONNX-exported model |
| Streamlit | Demo UI — interactive image upload and prediction |

---

## Dataset

[AI Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)  
Kaggle · `tristanzhang32` · Binary: `real` / `fake`  
Used for academic purposes only.

---

## Team

<!-- Add your names here -->
- [Name 1]
- [Name 2]
- [Name 3]
- [Name 4]

ZHAW School of Engineering · Machine Learning and Data in Operation
