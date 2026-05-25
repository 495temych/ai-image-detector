import argparse
import json
from pathlib import Path

import dagshub
import yaml
import torch
import mlflow
import mlflow.transformers
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from mlflow.tracking import MlflowClient


def export_onnx(
    model_name: str,
    output_dir: Path,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Download Production model from MLflow and export to ONNX.

    Returns (onnx_path, labels_path).
    """
    client = MlflowClient()
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise RuntimeError(f"No Production model found for '{model_name}'")
        run_id = versions[0].run_id

    model_uri = f"models:/{model_name}/Production"
    local_path = mlflow.artifacts.download_artifacts(model_uri)

    feature_extractor = AutoFeatureExtractor.from_pretrained(local_path)
    model = AutoModelForImageClassification.from_pretrained(local_path)
    model.eval()

    id2label = model.config.id2label
    labels_path = output_dir / "labels.json"
    labels_path.write_text(json.dumps({str(k): v for k, v in id2label.items()}))

    dummy_image = Image.new("RGB", (224, 224))
    inputs = feature_extractor(images=dummy_image, return_tensors="pt")
    dummy_pixel_values = inputs["pixel_values"]

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy_pixel_values,
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
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dagshub.init(repo_owner="marcosncosta1", repo_name="ai-image-detector", mlflow=True)
    export_onnx(
        model_name=config["mlflow"]["model_name"],
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
