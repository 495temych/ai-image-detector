"""Export the Production EfficientNet-B0 model to ONNX.

Loads the best_weights.pt from DVC (already pulled by the CI workflow),
exports to ONNX, and logs the artefact back to the MLflow run so the
Docker build step can pick it up as a CI artefact.
"""
import argparse
import json
from pathlib import Path


def export_onnx(
    weights_path: Path,
    output_dir: Path,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Export EfficientNet-B0 weights to ONNX and log to MLflow.

    Args:
        weights_path: path to best_weights.pt (from DVC pull).
        output_dir:   directory to write model.onnx and labels.json.
        run_id:       MLflow run to attach the artefact to (optional).

    Returns:
        (onnx_path, labels_path)
    """
    import mlflow                      # noqa: PLC0415
    import torch                       # noqa: PLC0415
    import torchvision.models as tvm  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── rebuild model architecture (must match training) ──────────────────
    # The Kaggle notebook replaced only classifier[1] (the default 1000-class
    # Linear) with a Sequential(Dropout, Linear(1280→1)), leaving classifier[0]
    # (the original Dropout) intact.  This produces keys:
    #   classifier.0.*  → original Dropout  (no params)
    #   classifier.1.0  → added Dropout     (no params)
    #   classifier.1.1  → Linear(1280, 1)   (weight + bias)
    model = tvm.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(1280, 1),
    )
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # ── labels ───────────────────────────────────────────────────────────
    labels = {"0": "real", "1": "fake"}
    labels_path = output_dir / "labels.json"
    labels_path.write_text(json.dumps(labels))

    # ── ONNX export ───────────────────────────────────────────────────────
    dummy = torch.zeros(1, 3, 224, 224)
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model → {onnx_path}  ({onnx_path.stat().st_size // 1024} KB)")
    print(f"Labels     → {labels_path}")

    # ── log artefact to MLflow run ────────────────────────────────────────
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(onnx_path),   artifact_path="onnx")
            mlflow.log_artifact(str(labels_path), artifact_path="onnx")
        print(f"Artefacts logged to MLflow run {run_id}")

    return onnx_path, labels_path


def main() -> None:
    import dagshub  # noqa: PLC0415
    import yaml     # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Export EfficientNet-B0 to ONNX")
    parser.add_argument("--config",      default="configs/eval_config.yaml")
    parser.add_argument("--weights",     default="models/efficientnet_v1/best_weights.pt",
                        help="Path to best_weights.pt (from dvc pull models)")
    parser.add_argument("--output-dir",  default="api/")
    parser.add_argument("--run-id",      default=None,
                        help="MLflow run ID to attach artefacts to")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)  # noqa: F841 (kept for future use)

    dagshub.init(repo_owner="marcosncosta1", repo_name="ai-image-detector", mlflow=True)

    export_onnx(
        weights_path=Path(args.weights),
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
