"""Register and promote model to Production in the MLflow Model Registry.

Strategy
--------
1. Read the run_id from evaluate.py (the Kaggle training run with real metrics).
2. Gate: accuracy must be >= threshold (double-check after evaluate already gated).
3. Promotion:
   a. Try to register a NEW version from the run's logged artifact
      (`runs:/{run_id}/model`).  This works if the Kaggle notebook called
      `mlflow.pytorch.log_model()` or equivalent.
   b. If no model artifact exists on that run (common when only metrics were
      logged from Kaggle), fall back to promoting the latest EXISTING registered
      version.  The weights themselves live in DVC; the registry version is just
      a pointer that CI uses as the promotion gate.
4. Transition the chosen version to Production, archiving all others.
"""
import argparse
import sys

try:
    from mlops_utils import ACCURACY_ALIASES, extract_accuracy  # script mode
except ImportError:
    from src.mlops_utils import ACCURACY_ALIASES, extract_accuracy  # test/package mode


# ---------------------------------------------------------------------------
# Pure helper — unit-testable with no ML dependencies
# ---------------------------------------------------------------------------

def should_register(accuracy: float, threshold: float) -> bool:
    return accuracy >= threshold


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _latest_existing_version(client, model_name: str):
    """Return the highest-numbered registered version, or None."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return sorted(versions, key=lambda v: int(v.version))[-1]


def promote_model(run_id: str, model_name: str, client) -> str:
    """Register (if possible) and promote the model to Production.

    Returns the promoted version string.
    """
    import mlflow  # noqa: PLC0415

    mv = None

    # ── try to register a new version from the run's model artifact ──────
    try:
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        print(f"Registered new version {mv.version} from run {run_id}")
    except Exception as exc:  # MlflowException if artifact not found
        print(
            f"\n⚠️  Could not register from run artifact:\n"
            f"   {exc}\n"
            f"   Falling back to promoting the latest existing registered version..."
        )
        mv = _latest_existing_version(client, model_name)
        if mv is None:
            print(
                "❌ No registered versions found and run has no model artifact.\n"
                "   Log the model from your Kaggle notebook with:\n"
                "     mlflow.pytorch.log_model(model, 'model')\n"
                "   then retrain and re-run this pipeline."
            )
            sys.exit(1)
        print(f"   Using existing version {mv.version}")

    # ── promote to Production ─────────────────────────────────────────────
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"\n✅ Promoted {model_name} v{mv.version} → Production")
    return mv.version


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import dagshub                             # noqa: PLC0415
    import yaml                               # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Gate on accuracy and promote the model to Production."
    )
    parser.add_argument("--config",  default="configs/eval_config.yaml")
    parser.add_argument("--run-id",  required=True,
                        help="MLflow run ID written by evaluate.py")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dagshub.init(repo_owner="marcosncosta1", repo_name="ai-image-detector", mlflow=True)
    client = MlflowClient()

    run       = client.get_run(args.run_id)
    accuracy  = extract_accuracy(run.data.metrics)
    if accuracy is None:
        keys = ", ".join(ACCURACY_ALIASES)
        print(f"❌ Run {args.run_id} has no accuracy metric (tried: {keys}).")
        sys.exit(1)
    threshold = config["model"]["accuracy_threshold"]
    model_name = config["mlflow"]["model_name"]

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Threshold: {threshold}")

    if not should_register(accuracy, threshold):
        print(f"\nFAIL: accuracy {accuracy:.4f} below threshold {threshold}.")
        sys.exit(1)

    promote_model(run_id=args.run_id, model_name=model_name, client=client)


if __name__ == "__main__":
    main()
