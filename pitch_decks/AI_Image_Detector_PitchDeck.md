---
marp: true
theme: default
paginate: true
size: 16:9
title: AI Image Authenticity Detector — MLOps Pipeline
---

<!--
Render to PDF with:
  npx @marp-team/marp-cli@latest AI_Image_Detector_PitchDeck.md --pdf
Or paste each slide (separated by "---") into PowerPoint / Google Slides / Keynote.
-->

# AI Image Authenticity Detector

## An MLOps Pipeline for AI-Generated vs. Real Image Classification

**Team:** [Name 1], [Name 2], [Name 3], [Name 4]
Machine Learning and Data in Operation · ZHAW School of Engineering
Spring 2026

---

## Project Goal & Motivation

**Context** — Generative models produce increasingly photorealistic images.
Telling real from synthetic has become an open trust and provenance problem
(journalism, evidence, identity, content moderation).

**Goal** — Build a **reproducible MLOps pipeline** that classifies an input
image as `real` or `ai-generated` with a confidence score.

**Scope** — The deliverable is the **pipeline**, not state-of-the-art accuracy.
We focus on versioning data, tracking every training run, registering models,
and serving them through an automated, reproducible workflow.

---

## Data

**Source:** [AI Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images) — Kaggle · `tristanzhang32`

- Binary labels: `real` / `fake`
- Train / validation / test splits tracked with **DVC**
- Each trained model is pinned to an exact dataset snapshot
- Dataset used for academic purposes only

> Future iterations: extend with newly labeled samples via a human-in-the-loop
> feedback stage to keep the model current as generator models evolve.

---

## MLOps Pipeline

```
 Kaggle        DVC               PyTorch           MLflow
 ──────▶ raw ─▶ versioned data ─▶ training ──▶ metrics · params · artifacts · registry
                                     │
                                     ▼
                         GitHub Actions  (CI: train → evaluate → register)
                                     │
                                     ▼
                    ONNX export ─▶ FastAPI + Docker ─▶ Streamlit demo UI
```

Every entry in the model registry is linked back to the **code commit** that
trained it and the **DVC-tracked data version** it was trained on.

---

## Components & Tools

| Stage | Tool | Role |
|-------|------|------|
| Data & code versioning | **DVC + Git** | Pin each dataset snapshot to a commit |
| Experiment tracking | **MLflow** | Metrics, params, artifacts, model registry |
| Training & export | **PyTorch + ONNX** | Fine-tune EfficientNet-B0, export portable model |
| Automation / CI | **GitHub Actions** | Train → evaluate → register on push to `main` |
| Serving | **FastAPI + Docker** | Containerized REST endpoint |
| Demo UI | **Streamlit** | Interactive image upload + prediction |

---

## Why This Is MLOps (not just ML)

- **Reproducibility** — any historical run can be rebuilt from its commit + DVC hash
- **Traceability** — MLflow links metrics ↔ params ↔ artifact ↔ commit ↔ data version
- **Automation** — CI pipeline retrains, evaluates, and registers on every merge
- **Deployability** — ONNX + Docker make the model portable and production-ready
- **Iterability** — new data → re-version → retrain → re-register, no manual glue

**Next steps:** configure the DVC remote, stand up the MLflow tracking server,
wire the GitHub Actions workflow, containerize the FastAPI service.
