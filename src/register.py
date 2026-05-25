import argparse
import sys


def should_register(accuracy: float, threshold: float) -> bool:
    return accuracy >= threshold


def promote_model(run_id: str, model_name: str) -> str:
    """Register the run's model artifact and promote to Production.
    Returns the new model version string."""
    import mlflow                          # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

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
    # Heavy MLOps imports — only needed when running the full pipeline
    import dagshub                             # noqa: PLC0415
    import yaml                                # noqa: PLC0415
    from mlflow.tracking import MlflowClient   # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dagshub.init(repo_owner="marcosncosta1", repo_name="ai-image-detector", mlflow=True)
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
