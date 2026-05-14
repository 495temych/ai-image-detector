# AI Image Detector — Project Summary

**What we're building:** A complete MLOps pipeline that takes an image and tells you whether it's real or AI-generated. The deliverable is the **pipeline**, not the model — we're demonstrating how a machine learning system gets versioned, tracked, automated, and deployed.

---

## How It Works (End to End)

1. **Dataset** — We download a subset (~500 images/class) of the [Kaggle AI vs. Real dataset](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images). Instead of storing images in Git, we use **DVC** to version them — every run is pinned to an exact dataset snapshot stored on **DagsHub**.

2. **Model** — We fine-tune **EfficientNet** (torchvision) on the Kaggle dataset using a [Kaggle GPU notebook](https://www.kaggle.com/code/marcosncosta/ai-vs-real-image-efficientnet-fine-tuning-86380e). EfficientNet was chosen because it was pre-trained on larger images and scales well with image resolution, giving better feature extraction for high-resolution photographs. The trained `.pt` weights are downloaded from Kaggle and versioned via DVC.

3. **Evaluation** — A script runs the model against our test split, computes accuracy / F1 / AUC-ROC, and logs everything — metrics, parameters, git commit SHA, and DVC data hash — to **MLflow** hosted on DagsHub. This makes every run fully traceable.

4. **Model Registry** — If accuracy passes a threshold (≥ 90%), the model gets registered in the **MLflow model registry** and promoted to `Production`. This is the gate between evaluation and deployment.

5. **ONNX Export** — The Production model gets exported to ONNX format (framework-agnostic, fast inference). The ONNX file gets uploaded back to MLflow as an artifact.

6. **CI/CD** — Steps 3–5 above run automatically via **GitHub Actions** on every push to `main` (4 sequential jobs: evaluate → register → export → docker build). The resulting Docker image gets pushed to GitHub Container Registry (GHCR).

7. **Serving** — A **FastAPI** app loads the ONNX model and exposes a single `POST /predict` endpoint. It runs inside a **Docker** container pulled from GHCR.

8. **Demo UI** — A **Streamlit** app lets you upload an image, calls the FastAPI endpoint, and shows the result (`REAL` or `AI-GENERATED`) with a confidence score.

---

## Tools at a Glance

| Tool | Role |
|---|---|
| **DVC** | Version the dataset and model weights — every run links back to the exact data and model it used |
| **DagsHub** | Hosts the DVC remote storage and the MLflow tracking server (free) |
| **MLflow** | Tracks every evaluation run (metrics, params, artifacts) + model registry |
| **PyTorch + Kaggle** | Fine-tune EfficientNet on Kaggle GPU; weights downloaded and versioned via DVC |
| **ONNX** | Framework-agnostic model export for fast, portable inference |
| **GitHub Actions** | Automates the evaluate → register → export → docker pipeline on every push |
| **FastAPI + Docker** | Containerised REST API serving the ONNX model |
| **Streamlit** | Interactive demo UI for the live presentation |

---

## What Already Exists

- Project concept, README, and website (`site/`) ✅
- Pitch deck ✅
- Design spec (`docs/superpowers/specs/2026-04-25-ai-image-detector-design.md`) ✅
- Implementation plan (`docs/superpowers/plans/2026-04-25-ai-image-detector-plan.md`) ✅

## What We're Building

All the code: dataset prep script, evaluation script, MLflow logging, model registration, ONNX export, FastAPI service, Dockerfile, Streamlit UI, and GitHub Actions workflow — 10 tasks, each with step-by-step instructions and exact code.

---

## Demo Day

We run two things locally:

1. The **Docker container** (pre-built by CI) serving the model via FastAPI
2. The **Streamlit app** calling that container

We upload 6–8 pre-selected images (clearly real, clearly AI, a few ambiguous) and show live predictions with confidence scores. Everything else — MLflow run history, model registry, GitHub Actions logs, DVC data versioning — gets shown on slides as screenshots.

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Model architecture | EfficientNet (fine-tuned) | Pre-trained on larger images; better feature extraction for high-resolution photos than ViT-224 |
| Training environment | Kaggle GPU notebook | Free GPU, no local setup; training happens once and weights are versioned via DVC |
| Dataset subset | ~500 images/class (~100MB) | Enough for meaningful metrics; fits DagsHub free tier |
| DVC remote | DagsHub (migrate to GCS when credits arrive) | Free, zero infra, MLflow in same place |
| CI training step | Replaced by eval + register | Training is done externally on Kaggle; CI only evaluates, registers, exports, and packages |
| Demo approach | Streamlit → FastAPI → ONNX | Shows serving layer live; MLflow/DVC shown on slides |

---

*ZHAW School of Engineering · Machine Learning and Data in Operation · Spring 2026*
*Team: Marcos, Artemi, Afshin, Jibin*
