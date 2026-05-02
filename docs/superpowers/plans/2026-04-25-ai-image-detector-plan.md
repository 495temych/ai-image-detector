# AI Image Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete MLOps pipeline that evaluates a fine-tuned EfficientNet classifier, registers it in MLflow, exports it to ONNX, containerises it in Docker, and serves it via FastAPI + Streamlit.

**Architecture:** EfficientNet is fine-tuned on Kaggle ([notebook](https://www.kaggle.com/code/marcosncosta/ai-vs-real-image-efficientnet-fine-tuning-86380e)), weights downloaded and versioned via DVC on DagsHub. CI evaluates those weights on a DVC-versioned test subset, logs metrics to MLflow on DagsHub, promotes to Production, exports to ONNX, and builds a Docker image. A Streamlit UI calls the FastAPI `/predict` endpoint. GitHub Actions automates the full evaluate→register→export→docker pipeline on every push to `main`.

**Tech Stack:** Python 3.11, PyTorch, torchvision (EfficientNet), ONNX/ONNXRuntime, MLflow, DVC, DagsHub, FastAPI, Uvicorn, Docker, Streamlit, GitHub Actions, scikit-learn, Pillow

---

## File Map

| File | Responsibility |
|---|---|
| `env.yaml` | Conda environment (local dev) |
| `requirements.txt` | Pip dependencies (CI + Docker) |
| `configs/eval_config.yaml` | All tuneable values: model path, threshold, data paths |
| `src/data/prepare_dataset.py` | Download Kaggle subset, create train/val/test splits |
| `src/evaluate.py` | Load EfficientNet `.pt`, run on test split, log metrics + model to MLflow |
| `src/register.py` | Check threshold, promote model to Production in MLflow registry |
| `src/export_onnx.py` | Load Production model from MLflow, export to ONNX + labels.json |
| `tests/test_evaluate.py` | Unit tests for metric computation |
| `tests/test_register.py` | Unit tests for threshold/promotion logic |
| `tests/test_api.py` | Integration tests for FastAPI endpoint |
| `api/main.py` | FastAPI app: loads ONNX at startup, exposes POST /predict |
| `api/requirements.txt` | Minimal deps for the Docker image |
| `api/Dockerfile` | Containerise FastAPI + ONNX model |
| `app/app.py` | Streamlit UI: upload image → call /predict → show result |
| `app/demo_images/` | 6–8 pre-selected images for the live demo |
| `.dvc/config` | DVC remote pointing to DagsHub |
| `.github/workflows/ci.yml` | CI: evaluate → register → export → docker |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `env.yaml`
- Create: `requirements.txt`
- Create: `configs/eval_config.yaml`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create the conda environment file**

Create `env.yaml`:

```yaml
name: mlops-img
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      - torch==2.3.0
      - torchvision==0.18.0
      - onnx==1.16.0
      - onnxruntime==1.18.0
      - mlflow==2.13.0
      - dvc[http]==3.50.0
      - scikit-learn==1.5.0
      - Pillow==10.3.0
      - numpy==1.26.4
      - pyyaml==6.0.1
      - fastapi==0.111.0
      - uvicorn==0.30.0
      - python-multipart==0.0.9
      - streamlit==1.36.0
      - requests==2.32.3
      - pytest==8.2.0
      - httpx==0.27.0
      - kaggle==1.6.14
```

- [ ] **Step 2: Create requirements.txt for CI and Docker base**

Create `requirements.txt`:

```
torch==2.3.0
torchvision==0.18.0
onnx==1.16.0
onnxruntime==1.18.0
mlflow==2.13.0
dvc[http]==3.50.0
scikit-learn==1.5.0
Pillow==10.3.0
numpy==1.26.4
pyyaml==6.0.1
fastapi==0.111.0
uvicorn==0.30.0
python-multipart==0.0.9
streamlit==1.36.0
requests==2.32.3
pytest==8.2.0
httpx==0.27.0
kaggle==1.6.14
```

- [ ] **Step 3: Create configs/eval_config.yaml**

Create `configs/eval_config.yaml`:

```yaml
model:
  model_path: "model/efficientnet.pt"   # path to fine-tuned weights (DVC-tracked)
  accuracy_threshold: 0.80
  input_size: 224

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  test_dir: "data/processed/test"
  per_class: 500
  split: [0.70, 0.15, 0.15]
  random_seed: 42

mlflow:
  experiment_name: "ai-image-detector"
  model_name: "ai-image-detector"

api:
  port: 8000
  model_path: "model.onnx"
  labels_path: "labels.json"
```

- [ ] **Step 4: Create placeholder directories and init files**

```bash
mkdir -p data/raw data/processed src/data tests app/demo_images api
touch data/raw/.gitkeep data/processed/.gitkeep
touch src/__init__.py src/data/__init__.py tests/__init__.py
```

- [ ] **Step 5: Update .gitignore**

Add to `.gitignore` (append, do not replace existing content):

```
# Data (DVC-managed)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/

# MLflow artifacts
mlruns/
mlflow_run_id.txt

# ONNX artifacts
*.onnx
labels.json

# Environment
.env
*.env

# macOS
.DS_Store
```

- [ ] **Step 6: Install environment and verify**

```bash
conda env create -f env.yaml
conda activate mlops-img
python -c "import torch, torchvision, mlflow, dvc; print('OK')"
```

Expected output: `OK`

- [ ] **Step 7: Commit scaffold**

```bash
git add env.yaml requirements.txt configs/ src/ tests/ data/ .gitignore
git commit -m "feat: project scaffold, config, and environment"
```

---

## Task 2: DagsHub + DVC Setup

**Files:**
- Create: `.dvc/config`
- Create: `.dvc/.gitignore`

> **Note:** This task requires one team member to complete the DagsHub web setup first. Only do this once. Other team members just run `dvc pull` after.

- [ ] **Step 1: Create DagsHub project**

1. Go to https://dagshub.com → click **New Project**
2. Choose **Connect a repository**
3. Select the `ai-image-detector` GitHub repo
4. DagsHub will show you two URLs — save them:
   - **DVC remote:** `https://dagshub.com/<your-username>/ai-image-detector.dvc`
   - **MLflow tracking:** `https://dagshub.com/<your-username>/ai-image-detector.mlflow`
5. Go to **Settings → Tokens** → generate a token → copy it

- [ ] **Step 2: Configure DVC remote**

Replace `<your-username>` with your actual DagsHub username:

```bash
dvc init
dvc remote add dagshub https://dagshub.com/<your-username>/ai-image-detector.dvc
dvc remote default dagshub
```

This creates `.dvc/config`. Verify the content:

```bash
cat .dvc/config
```

Expected output:
```ini
[core]
    remote = dagshub
[remote "dagshub"]
    url = https://dagshub.com/<your-username>/ai-image-detector.dvc
```

- [ ] **Step 3: Set local DagsHub credentials (not committed)**

```bash
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user <your-username>
dvc remote modify --local dagshub password <your-dagshub-token>
```

This writes to `.dvc/config.local` which is already git-ignored by DVC.

- [ ] **Step 4: Set MLflow env var in your shell profile**

Add to `~/.zshrc` (or `~/.bashrc`):

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<your-username>/ai-image-detector.mlflow"
export MLFLOW_TRACKING_USERNAME="<your-username>"
export MLFLOW_TRACKING_PASSWORD="<your-dagshub-token>"
```

Then reload: `source ~/.zshrc`

- [ ] **Step 5: Add GitHub Actions secrets**

Go to GitHub repo → **Settings → Secrets and variables → Actions** → add:

| Name | Value |
|---|---|
| `DAGSHUB_TOKEN` | Your DagsHub token |
| `DAGSHUB_USERNAME` | Your DagsHub username |

- [ ] **Step 6: Commit DVC config**

```bash
git add .dvc/config .dvc/.gitignore
git commit -m "feat: configure DVC remote (DagsHub)"
```

---

## Task 3: Dataset Preparation Script

**Files:**
- Create: `src/data/prepare_dataset.py`

- [ ] **Step 1: Install Kaggle API credentials**

Place your `kaggle.json` at `~/.kaggle/kaggle.json` (download from Kaggle → Account → API → Create New Token).

```bash
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list --search "tristanzhang32"
```

Expected output: shows `tristanzhang32/ai-generated-images-vs-real-images`

- [ ] **Step 2: Write the prepare_dataset.py script**

Create `src/data/prepare_dataset.py`:

```python
import argparse
import random
import shutil
from pathlib import Path


def prepare(raw_dir: Path, processed_dir: Path, per_class: int,
            split: list[float], seed: int) -> None:
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    random.seed(seed)

    class_names = ["real", "fake"]
    split_names = ["train", "val", "test"]

    for split_name in split_names:
        for cls in class_names:
            (processed_dir / split_name / cls).mkdir(parents=True, exist_ok=True)

    for cls in class_names:
        cls_dir = raw_dir / cls
        images = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        if len(images) < per_class:
            raise ValueError(
                f"Only {len(images)} images found in {cls_dir}, need {per_class}"
            )
        sampled = random.sample(images, per_class)

        n = len(sampled)
        n_train = int(n * split[0])
        n_val = int(n * split[1])

        assignments = (
            [("train", img) for img in sampled[:n_train]]
            + [("val", img) for img in sampled[n_train : n_train + n_val]]
            + [("test", img) for img in sampled[n_train + n_val :]]
        )

        for split_name, img_path in assignments:
            dest = processed_dir / split_name / cls / img_path.name
            shutil.copy2(img_path, dest)

        counts = {s: sum(1 for s2, _ in assignments if s2 == s) for s in split_names}
        print(f"  {cls}: train={counts['train']} val={counts['val']} test={counts['test']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--split", default="0.70/0.15/0.15")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split = [float(x) for x in args.split.split("/")]
    prepare(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        per_class=args.per_class,
        split=split,
        seed=args.seed,
    )
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Download dataset and run the script**

```bash
# Download (this pulls the full 52GB zip — delete src after running the script)
kaggle datasets download tristanzhang32/ai-generated-images-vs-real-images \
  -p data/raw --unzip

# Run split script
python src/data/prepare_dataset.py \
  --per-class 500 --split 0.70/0.15/0.15 --seed 42
```

Expected output:
```
  real: train=350 val=75 test=75
  fake: train=350 val=75 test=75
Done.
```

- [ ] **Step 4: Delete the raw Kaggle download (keep only the processed subset)**

```bash
# Remove everything in data/raw except the .gitkeep
find data/raw -type f ! -name '.gitkeep' -delete
find data/raw -mindepth 1 -type d -empty -delete
```

- [ ] **Step 5: Track with DVC and push to DagsHub**

```bash
dvc add data/processed
git add data/processed.dvc .gitignore
git commit -m "feat: add DVC-tracked dataset splits (500/class)"
dvc push
```

Expected: DVC uploads ~100MB to DagsHub. Confirm at `https://dagshub.com/<your-username>/ai-image-detector` → Files tab.

- [ ] **Step 6: Commit the script**

```bash
git add src/data/prepare_dataset.py
git commit -m "feat: dataset preparation script"
```

---

## Task 4: Evaluation Script

**Files:**
- Create: `src/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests first**

Create `tests/test_evaluate.py`:

```python
import pytest
from src.evaluate import compute_metrics


def test_compute_metrics_perfect():
    true_labels = [0, 0, 1, 1]
    pred_labels = [0, 0, 1, 1]
    pred_scores = [0.05, 0.05, 0.95, 0.95]
    m = compute_metrics(true_labels, pred_labels, pred_scores)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["auc_roc"] == 1.0


def test_compute_metrics_all_wrong():
    true_labels = [0, 0, 1, 1]
    pred_labels = [1, 1, 0, 0]
    pred_scores = [0.95, 0.95, 0.05, 0.05]
    m = compute_metrics(true_labels, pred_labels, pred_scores)
    assert m["accuracy"] == 0.0


def test_compute_metrics_returns_float():
    true_labels = [0, 1, 0, 1]
    pred_labels = [0, 1, 1, 0]
    pred_scores = [0.2, 0.8, 0.6, 0.4]
    m = compute_metrics(true_labels, pred_labels, pred_scores)
    assert isinstance(m["accuracy"], float)
    assert isinstance(m["f1"], float)
    assert isinstance(m["auc_roc"], float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_evaluate.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `src.evaluate` doesn't exist yet.

- [ ] **Step 3: Write evaluate.py**

Create `src/evaluate.py`:

```python
import os
import subprocess
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as T
import mlflow
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip()


def get_dvc_hash(dvc_file: str = "data/processed.dvc") -> str:
    with open(dvc_file) as f:
        info = yaml.safe_load(f)
    return info["outs"][0]["md5"]


def load_model(model_path: str) -> torch.nn.Module:
    model = tv_models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def get_transform() -> T.Compose:
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    pred_scores: list[float],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "f1": float(f1_score(true_labels, pred_labels)),
        "auc_roc": float(roc_auc_score(true_labels, pred_scores)),
    }


def load_test_images(test_dir: Path) -> tuple[list[str], list[int]]:
    """Returns (image_paths, labels) where label 0=real, 1=fake."""
    images, labels = [], []
    label_map = {"real": 0, "fake": 1}
    for class_name, label in label_map.items():
        class_dir = test_dir / class_name
        for img_path in sorted(class_dir.glob("*.jpg")) + sorted(
            class_dir.glob("*.png")
        ):
            images.append(str(img_path))
            labels.append(label)
    return images, labels


def run_evaluation(config: dict) -> dict:
    model_path = config["model"]["model_path"]
    test_dir = Path(config["data"]["test_dir"])

    model = load_model(model_path)
    transform = get_transform()
    images, true_labels = load_test_images(test_dir)

    pred_labels, pred_scores = [], []
    with torch.no_grad():
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            logits = model(tensor)[0]
            probs = torch.softmax(logits, dim=0).numpy()
            pred_idx = int(np.argmax(probs))
            pred_labels.append(pred_idx)
            pred_scores.append(float(probs[1]))  # prob of class 1 (fake)

    return compute_metrics(true_labels, pred_labels, pred_scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        mlflow.log_param("model_path", config["model"]["model_path"])
        mlflow.log_param("threshold", config["model"]["accuracy_threshold"])
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dvc_hash", get_dvc_hash())

        metrics = run_evaluation(config)
        mlflow.log_metrics(metrics)

        mlflow.log_artifact(config["model"]["model_path"], artifact_path="model")

        print(f"Run ID: {run.info.run_id}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        with open("mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_evaluate.py -v
```

Expected output:
```
tests/test_evaluate.py::test_compute_metrics_perfect PASSED
tests/test_evaluate.py::test_compute_metrics_all_wrong PASSED
tests/test_evaluate.py::test_compute_metrics_returns_float PASSED
3 passed
```

- [ ] **Step 5: Place the trained EfficientNet weights**

Download the `.pt` file from the Kaggle notebook output and place it at the path in config (default `model/efficientnet.pt`):

```bash
mkdir -p model
# copy the .pt file from your Kaggle download here
```

> Track the weights with DVC so every run is pinned to a specific model version:
>
> ```bash
> dvc add model/efficientnet.pt
> git add model/efficientnet.pt.dvc .gitignore
> git commit -m "feat: add DVC-tracked EfficientNet weights"
> dvc push
> ```

- [ ] **Step 6: Run evaluation locally against the test split**

Ensure `MLFLOW_TRACKING_URI` is set, then:

```bash
python src/evaluate.py --config configs/eval_config.yaml
```

Expected output (values will vary):
```
Run ID: <some-uuid>
  accuracy: 0.8600
  f1: 0.8612
  auc_roc: 0.9201
```

Open `https://dagshub.com/<your-username>/ai-image-detector.mlflow` and confirm the run appears.

- [ ] **Step 7: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation script with MLflow logging"
```

---

## Task 5: Model Registration Script

**Files:**
- Create: `src/register.py`
- Create: `tests/test_register.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_register.py`:

```python
import pytest
from src.register import should_register


def test_should_register_above_threshold():
    assert should_register(accuracy=0.95, threshold=0.90) is True


def test_should_register_below_threshold():
    assert should_register(accuracy=0.85, threshold=0.90) is False


def test_should_register_at_threshold():
    assert should_register(accuracy=0.90, threshold=0.90) is True


def test_should_register_zero_accuracy():
    assert should_register(accuracy=0.0, threshold=0.90) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_register.py -v
```

Expected: `ImportError` — `src.register` doesn't exist yet.

- [ ] **Step 3: Write register.py**

Create `src/register.py`:

```python
import os
import argparse
import sys

import yaml
import mlflow
from mlflow.tracking import MlflowClient


def should_register(accuracy: float, threshold: float) -> bool:
    return accuracy >= threshold


def promote_model(run_id: str, model_name: str) -> str:
    """Register the run's model artifact and promote to Production.
    Returns the new model version string."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Registered {model_name} version {mv.version} → Production")
    return mv.version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    run = client.get_run(args.run_id)
    accuracy = run.data.metrics["accuracy"]
    threshold = config["model"]["accuracy_threshold"]

    print(f"Accuracy: {accuracy:.4f}  Threshold: {threshold}")

    if not should_register(accuracy, threshold):
        print(f"FAIL: accuracy {accuracy:.4f} below threshold {threshold}. Exiting.")
        sys.exit(1)

    promote_model(
        run_id=args.run_id,
        model_name=config["mlflow"]["model_name"],
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_register.py -v
```

Expected:
```
tests/test_register.py::test_should_register_above_threshold PASSED
tests/test_register.py::test_should_register_below_threshold PASSED
tests/test_register.py::test_should_register_at_threshold PASSED
tests/test_register.py::test_should_register_zero_accuracy PASSED
4 passed
```

- [ ] **Step 5: Run registration locally**

```bash
RUN_ID=$(cat mlflow_run_id.txt)
python src/register.py --config configs/eval_config.yaml --run-id $RUN_ID
```

Expected output:
```
Accuracy: 0.9467  Threshold: 0.90
Registered ai-image-detector version 1 → Production
```

Verify in the DagsHub MLflow UI: Models tab → `ai-image-detector` → version 1 is `Production`.

- [ ] **Step 6: Commit**

```bash
git add src/register.py tests/test_register.py
git commit -m "feat: model registration and MLflow promotion"
```

---

## Task 6: ONNX Export Script

**Files:**
- Create: `src/export_onnx.py`

- [ ] **Step 1: Write export_onnx.py**

Create `src/export_onnx.py`:

```python
import os
import argparse
import json
from pathlib import Path

import yaml
import torch
import torchvision.models as tv_models
import mlflow
from mlflow.tracking import MlflowClient


LABELS = {"0": "REAL", "1": "FAKE"}


def _build_model(pt_path: Path) -> torch.nn.Module:
    model = tv_models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(torch.load(str(pt_path), map_location="cpu"))
    model.eval()
    return model


def export_onnx(model_name: str, output_dir: Path) -> tuple[Path, Path]:
    """Download Production model artifact from MLflow, export to ONNX.

    Returns (onnx_path, labels_path).
    """
    client = MlflowClient()
    output_dir.mkdir(parents=True, exist_ok=True)

    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise RuntimeError(f"No Production model found for '{model_name}'")
    run_id = versions[0].run_id

    local_dir = Path(mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model"))
    pt_files = list(local_dir.glob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"No .pt file found in downloaded artifacts: {local_dir}")

    model = _build_model(pt_files[0])

    labels_path = output_dir / "labels.json"
    labels_path.write_text(json.dumps(LABELS))

    dummy = torch.zeros(1, 3, 224, 224)
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=14,
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
        mlflow.log_artifact(str(labels_path), artifact_path="onnx")

    print(f"ONNX model → {onnx_path}")
    print(f"Labels     → {labels_path}")
    return onnx_path, labels_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    export_onnx(
        model_name=config["mlflow"]["model_name"],
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run ONNX export locally**

```bash
python src/export_onnx.py --config configs/eval_config.yaml --output-dir .
```

Expected output:
```
ONNX model → model.onnx
Labels     → labels.json
```

- [ ] **Step 3: Verify ONNX output shape**

```bash
python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {'pixel_values': dummy})
print('Output shape:', outputs[0].shape)
assert outputs[0].shape == (1, 2), 'Expected (1, 2) logits'
print('PASS')
"
```

Expected:
```
Output shape: (1, 2)
PASS
```

- [ ] **Step 4: Commit**

```bash
git add src/export_onnx.py
git commit -m "feat: ONNX export from MLflow Production model"
```

---

## Task 7: FastAPI Service + Dockerfile

**Files:**
- Create: `api/main.py`
- Create: `api/requirements.txt`
- Create: `api/Dockerfile`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing API tests**

Create `tests/test_api.py`:

```python
import io
import numpy as np
import pytest
from unittest import mock
from PIL import Image
from fastapi.testclient import TestClient


def make_jpeg_bytes(color: tuple = (100, 150, 200)) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    """Patch the lazy helpers so no real model files are needed."""
    mock_session = mock.MagicMock()
    mock_session.run.return_value = [np.array([[0.1, 0.9]])]  # logits: REAL (idx 1) wins

    with (
        mock.patch("api.main._get_session", return_value=mock_session),
        mock.patch("api.main._get_id2label", return_value={"0": "FAKE", "1": "REAL"}),
    ):
        from api.main import app
        yield TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_valid_structure(client):
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ("real", "ai-generated")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_real_image(client):
    # Mock returns logits [0.1, 0.9] → REAL (index 1) wins
    r = client.post(
        "/predict",
        files={"file": ("img.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert r.json()["label"] == "real"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: `ModuleNotFoundError` — `api.main` doesn't exist yet.

- [ ] **Step 3: Write api/main.py**

Create `api/main.py`:

```python
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "model.onnx")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.json")

# ImageNet normalisation constants matching EfficientNet training preprocessing
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(title="AI Image Detector", version="1.0")

# Lazy-loaded singletons — initialised on first request, not at import time.
# This makes the module importable in tests without real model files.
_session = None
_id2label = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(MODEL_PATH)
    return _session


def _get_id2label() -> dict[str, str]:
    global _id2label
    if _id2label is None:
        _id2label = json.loads(Path(LABELS_PATH).read_text())
    return _id2label


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((256, 256))
    left = (256 - 224) // 2
    image = image.crop((left, left, left + 224, left + 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _predict(image: Image.Image) -> dict:
    session = _get_session()
    id2label = _get_id2label()

    pixel_values = _preprocess(image)
    logits = session.run(None, {"pixel_values": pixel_values})[0][0]
    probs = _softmax(logits)

    pred_idx = int(np.argmax(probs))
    raw_label = id2label[str(pred_idx)].upper()
    label = "real" if raw_label == "REAL" else "ai-generated"
    confidence = float(probs[pred_idx])
    return {"label": label, "confidence": round(confidence, 4)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = _predict(image)
    return JSONResponse(result)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected:
```
tests/test_api.py::test_health PASSED
tests/test_api.py::test_predict_returns_valid_structure PASSED
tests/test_api.py::test_predict_real_image PASSED
3 passed
```

- [ ] **Step 5: Create api/requirements.txt (minimal, for Docker)**

Create `api/requirements.txt`:

```
fastapi==0.111.0
uvicorn==0.30.0
onnxruntime==1.18.0
Pillow==10.3.0
numpy==1.26.4
python-multipart==0.0.9
```

- [ ] **Step 6: Create api/Dockerfile**

Create `api/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model artifacts copied in by CI (or manually for local builds)
COPY model.onnx .
COPY labels.json .

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 7: Build and run Docker locally**

```bash
# Copy model artifacts into api/ directory
cp model.onnx labels.json api/

# Build
docker build -t ai-image-detector-api api/

# Run
docker run -d -p 8000:8000 --name ai-api ai-image-detector-api

# Smoke test
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 8: Test the running container with a real image**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/processed/test/real/$(ls data/processed/test/real | head -1)"
```

Expected (values will vary): `{"label":"real","confidence":0.9834}`

- [ ] **Step 9: Commit**

```bash
git add api/ tests/test_api.py
git commit -m "feat: FastAPI service with ONNX inference and Dockerfile"
```

---

## Task 8: Streamlit Demo UI

**Files:**
- Create: `app/app.py`
- Create: `app/demo_images/` (manual — 6–8 images)

- [ ] **Step 1: Collect demo images**

Manually copy 6–8 images into `app/demo_images/`. Naming convention:

```
app/demo_images/
├── real_01.jpg       # clear real photo (landscape, person, etc.)
├── real_02.jpg       # clear real photo
├── ai_01.jpg         # obvious AI-generated (smooth skin, surreal)
├── ai_02.jpg         # obvious AI-generated
├── ambiguous_01.jpg  # hard to tell
├── ambiguous_02.jpg  # hard to tell
```

Source: use 2 images from `data/processed/test/real/`, 2 from `data/processed/test/fake/`, and find 2 ambiguous ones online. Commit these images to Git (they're small and for demo use only).

- [ ] **Step 2: Write app/app.py**

Create `app/app.py`:

```python
import os
import io
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEMO_DIR = Path(__file__).parent / "demo_images"

st.set_page_config(page_title="AI Image Detector", layout="centered")
st.title("AI Image Detector")
st.caption("Upload an image to find out if it's real or AI-generated.")

tab_upload, tab_demo = st.tabs(["Upload", "Demo Images"])

with tab_upload:
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="upload"
    )
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)
        with st.spinner("Analysing…"):
            try:
                r = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=10,
                )
                r.raise_for_status()
                result = r.json()
                label = result["label"]
                confidence = result["confidence"]
                if label == "real":
                    st.success(f"**REAL** — {confidence:.1%} confidence")
                else:
                    st.error(f"**AI-GENERATED** — {confidence:.1%} confidence")
                st.progress(confidence)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API at " + API_URL + ". Is Docker running?")

with tab_demo:
    if DEMO_DIR.exists():
        demo_files = sorted(DEMO_DIR.glob("*.jpg")) + sorted(DEMO_DIR.glob("*.png"))
        if demo_files:
            selected = st.selectbox(
                "Choose a demo image",
                options=demo_files,
                format_func=lambda p: p.name,
            )
            if selected:
                image = Image.open(selected)
                st.image(image, use_container_width=True)
                if st.button("Analyse"):
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    with st.spinner("Analysing…"):
                        try:
                            r = requests.post(
                                f"{API_URL}/predict",
                                files={"file": (selected.name, buf.getvalue(), "image/jpeg")},
                                timeout=10,
                            )
                            r.raise_for_status()
                            result = r.json()
                            label = result["label"]
                            confidence = result["confidence"]
                            if label == "real":
                                st.success(f"**REAL** — {confidence:.1%} confidence")
                            else:
                                st.error(f"**AI-GENERATED** — {confidence:.1%} confidence")
                            st.progress(confidence)
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach API. Is Docker running?")
        else:
            st.info("No demo images found in app/demo_images/")
    else:
        st.info("app/demo_images/ directory not found.")
```

- [ ] **Step 3: Run and verify Streamlit locally**

Ensure Docker container is running (`docker ps` → see `ai-api`), then:

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in a browser. Upload an image and confirm you get a label and confidence bar. Switch to the Demo Images tab and verify pre-loaded images work.

- [ ] **Step 4: Commit**

```bash
git add app/app.py app/demo_images/
git commit -m "feat: Streamlit demo UI with upload + demo image tabs"
```

---

## Task 9: GitHub Actions CI/CD

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create .github/workflows/ci.yml**

Create `.github/workflows/ci.yml`:

```yaml
name: MLOps Pipeline

on:
  push:
    branches: [main]

env:
  MLFLOW_TRACKING_URI: https://dagshub.com/${{ secrets.DAGSHUB_USERNAME }}/ai-image-detector.mlflow
  REGISTRY: ghcr.io

jobs:
  evaluate:
    runs-on: ubuntu-latest
    outputs:
      run_id: ${{ steps.run_eval.outputs.run_id }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Configure DVC remote credentials
        run: |
          dvc remote modify dagshub --local auth basic
          dvc remote modify dagshub --local user ${{ secrets.DAGSHUB_USERNAME }}
          dvc remote modify dagshub --local password ${{ secrets.DAGSHUB_TOKEN }}

      - name: Pull dataset and model weights from DagsHub
        run: |
          dvc pull data/processed.dvc
          dvc pull model/efficientnet.pt.dvc

      - name: Run evaluation
        id: run_eval
        env:
          MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
          MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
        run: |
          python src/evaluate.py --config configs/eval_config.yaml
          echo "run_id=$(cat mlflow_run_id.txt)" >> $GITHUB_OUTPUT

      - name: Upload run_id artifact
        uses: actions/upload-artifact@v4
        with:
          name: mlflow-run-id
          path: mlflow_run_id.txt

  register:
    needs: evaluate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: pip install -r requirements.txt

      - name: Download run_id
        uses: actions/download-artifact@v4
        with:
          name: mlflow-run-id

      - name: Register and promote model
        env:
          MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
          MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
        run: |
          python src/register.py \
            --config configs/eval_config.yaml \
            --run-id $(cat mlflow_run_id.txt)

  export:
    needs: register
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: pip install -r requirements.txt

      - name: Export ONNX model
        env:
          MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
          MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
        run: |
          python src/export_onnx.py \
            --config configs/eval_config.yaml \
            --output-dir .

      - name: Upload ONNX artifacts
        uses: actions/upload-artifact@v4
        with:
          name: model-artifacts
          path: |
            model.onnx
            labels.json

  docker:
    needs: export
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Download model artifacts
        uses: actions/download-artifact@v4
        with:
          name: model-artifacts
          path: api/

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: api/
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/ai-image-detector-api:latest
            ghcr.io/${{ github.repository_owner }}/ai-image-detector-api:${{ github.sha }}
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
echo "YAML valid"
```

Expected: `YAML valid`

- [ ] **Step 3: Commit and push to trigger CI**

```bash
git add .github/workflows/ci.yml
git commit -m "feat: GitHub Actions CI pipeline (eval → register → export → docker)"
git push origin main
```

- [ ] **Step 4: Monitor the workflow**

Go to GitHub → Actions tab → watch the `MLOps Pipeline` workflow run. Verify all 4 jobs go green: `evaluate`, `register`, `export`, `docker`.

If a job fails, check the log. Common issues:
- `evaluate` fails: check `DAGSHUB_TOKEN` / `DAGSHUB_USERNAME` secrets are set correctly
- `docker` fails on push: check repo **Packages** visibility (set to public or ensure token has `packages:write`)

- [ ] **Step 5: Pull the CI-built Docker image locally**

```bash
docker pull ghcr.io/<your-github-username>/ai-image-detector-api:latest
docker run -d -p 8000:8000 --name ai-api-ci \
  ghcr.io/<your-github-username>/ai-image-detector-api:latest
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

---

## Task 10: End-to-End Smoke Test + Demo Prep

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Full end-to-end demo run-through**

With the CI-built Docker container running:

```bash
# 1. Confirm API is live
curl http://localhost:8000/health

# 2. Start Streamlit
streamlit run app/app.py

# 3. Open http://localhost:8501
# 4. Go to Demo Images tab
# 5. Cycle through all 6-8 demo images, clicking Analyse on each
# 6. Confirm labels + confidence scores appear correctly
```

- [ ] **Step 3: Screenshot key pipeline components for slides**

Capture screenshots of:
1. DagsHub MLflow UI → Experiments → your run (shows accuracy, F1, AUC-ROC, git_commit tag, dvc_hash tag)
2. DagsHub MLflow UI → Models → `ai-image-detector` → version 1 → stage: Production
3. GitHub Actions → the passing workflow run showing all 4 green jobs
4. DagsHub → Files tab showing `data/processed.dvc` (data versioning proof)
5. GHCR → the published Docker image (`ghcr.io/...`)
6. Streamlit demo showing a REAL result
7. Streamlit demo showing an AI-GENERATED result

- [ ] **Step 4: Final commit**

```bash
git add app/demo_images/
git commit -m "chore: add demo images for presentation"
git push origin main
```

---

## Running the Full Demo (Presentation Day)

```bash
# 1. Pull latest CI-built image
docker pull ghcr.io/<your-github-username>/ai-image-detector-api:latest

# 2. Start API container
docker run -d -p 8000:8000 --name demo-api \
  ghcr.io/<your-github-username>/ai-image-detector-api:latest

# 3. Start Streamlit
streamlit run app/app.py

# 4. Open http://localhost:8501 → Demo Images tab
```

**Fallback** (if Docker fails): set `USE_LOCAL_MODEL=1` env var — Streamlit loads ONNX directly without calling the API. Add this path in `app/app.py` Task 8 if needed.

---

## Spec Coverage Checklist

| Spec requirement | Task |
|---|---|
| Data versioning with DVC | Task 2, 3 |
| Model weights versioning with DVC | Task 4 |
| DagsHub remote (DVC + MLflow) | Task 2 |
| GCS migration path documented | Spec §9 (no code needed) |
| Evaluate fine-tuned EfficientNet on test split | Task 4 |
| Log metrics + model to MLflow | Task 4 |
| Register + promote to Production | Task 5 |
| ONNX export + artifact upload | Task 6 |
| FastAPI /predict endpoint | Task 7 |
| Docker containerisation | Task 7 |
| GHCR push | Task 9 |
| Streamlit UI | Task 8 |
| GitHub Actions CI (4 jobs) | Task 9 |
| Traceability triple (commit, DVC hash, run ID) | Task 4 |
| 6–8 demo images | Task 10 |
| Presentation screenshots | Task 10 |
