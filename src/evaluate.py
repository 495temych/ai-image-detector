import subprocess
import argparse
from pathlib import Path

import dagshub
import yaml
import mlflow
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Class indices confirmed from training sanity check
FAKE_IDX, REAL_IDX = 1, 0


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip()


def get_dvc_hash(dvc_file: str = "data/processed.dvc") -> str:
    try:
        with open(dvc_file) as f:
            info = yaml.safe_load(f)
        return info["outs"][0]["md5"]
    except (FileNotFoundError, KeyError, TypeError):
        return "unknown"


def _infer_onnx(session: ort.InferenceSession, img_path: str) -> tuple[int, float]:
    img = Image.open(img_path).convert("RGB")
    x = _TRANSFORM(img).unsqueeze(0).numpy()
    logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()
    pred = int(np.argmax(probs))
    return pred, float(probs[FAKE_IDX])


def compute_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    pred_scores: list[float],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "f1":       float(f1_score(true_labels, pred_labels)),
        "auc_roc":  float(roc_auc_score(true_labels, pred_scores)),
    }


def load_test_images(test_dir: Path) -> tuple[list[str], list[int]]:
    """Returns (image_paths, labels) where label 0=real, 1=fake."""
    images, labels = [], []
    label_map = {"real": REAL_IDX, "fake": FAKE_IDX}
    for class_name, label in label_map.items():
        class_dir = test_dir / class_name
        for img_path in sorted(class_dir.glob("*.jpg")) + sorted(
            class_dir.glob("*.png")
        ):
            images.append(str(img_path))
            labels.append(label)
    return images, labels


def run_evaluation(config: dict) -> dict:
    onnx_path = config["model"]["onnx_path"]
    test_dir  = Path(config["data"]["test_dir"])

    session = ort.InferenceSession(onnx_path)
    images, true_labels = load_test_images(test_dir)

    pred_labels, pred_scores = [], []
    for img_path in images:
        label, fake_prob = _infer_onnx(session, img_path)
        pred_labels.append(label)
        pred_scores.append(fake_prob)

    metrics = compute_metrics(true_labels, pred_labels, pred_scores)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "onnx_path": onnx_path,
            "fake_idx":  FAKE_IDX,
            "n_test":    len(images),
        })
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dvc_hash",   get_dvc_hash())
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(onnx_path, artifact_path="model")

        with open("mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID: {run.info.run_id}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dagshub.init(repo_owner="495temych", repo_name="ai-image-detector", mlflow=True)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    run_evaluation(config)


if __name__ == "__main__":
    main()
