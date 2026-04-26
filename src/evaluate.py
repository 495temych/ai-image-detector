import subprocess
import argparse
from pathlib import Path

import dagshub
import yaml
import mlflow
import mlflow.transformers
from transformers import pipeline, AutoConfig
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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


def parse_pipeline_output(
    result: list[dict], fake_label: str
) -> tuple[int, float]:
    """Convert HF pipeline output to (pred_label, fake_probability).

    pred_label: 1 if FAKE, 0 if REAL
    fake_probability: probability the image is fake (used for AUC-ROC)
    """
    top = result[0]
    is_fake = top["label"].upper() == fake_label.upper()
    if is_fake:
        return 1, float(top["score"])
    else:
        fake_score = next(
            (r["score"] for r in result if r["label"].upper() == fake_label.upper()),
            1.0 - top["score"],
        )
        return 0, float(fake_score)


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


def run_evaluation(config: dict) -> tuple[dict, object]:
    model_id = config["model"]["hf_model_id"]
    test_dir = Path(config["data"]["test_dir"])

    model_config = AutoConfig.from_pretrained(model_id)
    fake_label = next(
        v for v in model_config.id2label.values() if v.upper() == "FAKE"
    )

    classifier = pipeline("image-classification", model=model_id)
    images, true_labels = load_test_images(test_dir)

    pred_labels, pred_scores = [], []
    for img_path in images:
        result = classifier(img_path)
        label, score = parse_pipeline_output(result, fake_label=fake_label)
        pred_labels.append(label)
        pred_scores.append(score)

    metrics = compute_metrics(true_labels, pred_labels, pred_scores)
    return metrics, classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dagshub.init(repo_owner="495temych", repo_name="ai-image-detector", mlflow=True)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        mlflow.log_param("model_id", config["model"]["hf_model_id"])
        mlflow.log_param("threshold", config["model"]["accuracy_threshold"])
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dvc_hash", get_dvc_hash())

        metrics, classifier = run_evaluation(config)
        mlflow.log_metrics(metrics)

        mlflow.transformers.log_model(
            transformers_model=classifier,
            artifact_path="model",
        )

        print(f"Run ID: {run.info.run_id}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        with open("mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()
