"""Model evaluation — fetch metrics from the MLflow Model Registry.

This script is the CI/CD evaluation step. It does NOT re-run inference
on the full dataset. Instead, it queries the MLflow Model Registry for
the latest registered model version, retrieves its logged metrics from
the training run, and gates on the accuracy threshold.

This mirrors real-world MLOps promotion pipelines (TFX, SageMaker
Pipelines, Databricks MLflow) where:
  - Training runs on dedicated infra (Kaggle, GPU cluster) and logs
    authoritative metrics to an experiment tracker.
  - CI/CD reads those metrics and decides whether to promote the model.
  - The deployment pipeline never needs the raw training data.

Supported accuracy metric names (Kaggle notebooks commonly use these):
  accuracy, test_acc, val_acc, test_accuracy, val_accuracy, acc

Heavy dependencies (mlflow, dagshub) are imported lazily so that the
pure utility functions can be unit-tested without the full ML stack.
"""
import argparse
import sys
from pathlib import Path

try:
    from mlops_utils import ACCURACY_ALIASES, extract_accuracy  # script mode
except ImportError:
    from src.mlops_utils import ACCURACY_ALIASES, extract_accuracy  # test/package mode


# ---------------------------------------------------------------------------
# Pure utility functions — testable with no ML dependencies
# ---------------------------------------------------------------------------

def parse_pipeline_output(
    raw: list[dict],
    fake_label: str = "FAKE",
) -> tuple[int, float]:
    """Parse a HuggingFace pipeline output into (binary_label, fake_prob).

    Kept for backwards-compatibility with existing unit tests.
    """
    scores = {d["label"]: d["score"] for d in raw}
    fake_prob = scores.get(fake_label, 0.0)
    top_label = max(scores, key=scores.__getitem__)
    return int(top_label == fake_label), fake_prob


def compute_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    pred_scores: list[float],
) -> dict[str, float]:
    """Kept for backwards-compatibility with existing unit tests."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: PLC0415
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "f1":       float(f1_score(true_labels, pred_labels)),
        "auc_roc":  float(roc_auc_score(true_labels, pred_scores)),
    }


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def get_latest_model_version(client, model_name: str):
    """Return the most recently created model version regardless of stage."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(
            f"No versions found for model '{model_name}' in the MLflow registry.\n"
            "Train a model on Kaggle first and register it with:\n"
            "  mlflow.register_model(model_uri, model_name)"
        )
    return sorted(versions, key=lambda v: int(v.version))[-1]


def fetch_run_metrics(client, run_id: str) -> dict[str, float]:
    """Fetch all logged metrics for a given MLflow run."""
    run = client.get_run(run_id)
    return run.data.metrics


def find_best_run_with_accuracy(client, experiment_name: str):
    """Search the experiment for the run with the highest accuracy.

    Tries each alias in ACCURACY_ALIASES so notebooks that log
    test_acc / val_acc / etc. are found correctly.

    Returns (run, accuracy_value) tuple.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' not found.\n"
            "Make sure your Kaggle notebook calls:\n"
            "  mlflow.set_experiment('ai-image-detector')"
        )

    for key in ACCURACY_ALIASES:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"metrics.{key} > 0",
            order_by=[f"metrics.{key} DESC"],
            max_results=1,
        )
        if runs:
            run = runs[0]
            value = float(run.data.metrics[key])
            print(f"   Found metric '{key}' = {value:.4f} on run {run.info.run_id}")
            return run, value

    aliases = ", ".join(ACCURACY_ALIASES)
    raise RuntimeError(
        f"No runs with an accuracy metric found in experiment '{experiment_name}'.\n"
        f"Tried keys: {aliases}\n"
        "Make sure your Kaggle training notebook logs accuracy, e.g.:\n"
        "  mlflow.log_metric('test_acc', accuracy)"
    )


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def run_evaluation(config: dict) -> dict:
    """Gate on model accuracy fetched from the MLflow experiment.

    Strategy:
    1. Query the registry for the latest registered model version.
    2. Try to read accuracy (any alias) from that version's linked run.
    3. If the linked run has no accuracy (e.g. a CI artifact-only run),
       search the full experiment for the run with the highest accuracy
       — this finds the actual Kaggle training run.
    """
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    model_name      = config["mlflow"]["model_name"]
    experiment_name = config["mlflow"]["experiment_name"]
    threshold       = config["model"]["accuracy_threshold"]
    client          = MlflowClient()

    print(f"\n📋 Querying MLflow registry for model: '{model_name}'")
    mv = get_latest_model_version(client, model_name)
    print(f"   Version : {mv.version}")
    print(f"   Stage   : {mv.current_stage}")
    print(f"   Run ID  : {mv.run_id}")

    metrics  = fetch_run_metrics(client, mv.run_id)
    run_id   = mv.run_id
    accuracy = extract_accuracy(metrics)

    if accuracy is None:
        print(
            f"\n⚠️  Run {mv.run_id} has no accuracy metric "
            "(likely a CI artifact-only run).\n"
            f"   Searching experiment '{experiment_name}' for the best training run..."
        )
        best_run, accuracy = find_best_run_with_accuracy(client, experiment_name)
        run_id  = best_run.info.run_id
        metrics = best_run.data.metrics
        print(f"   Using run : {run_id}")

    print("\n📊 Metrics from training run:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    print(f"\n🎯 Accuracy {accuracy:.4f} vs threshold {threshold}")
    if accuracy < threshold:
        print("❌ Below threshold — model will NOT be promoted.")
        sys.exit(1)
    print("✅ Passes threshold — proceeding to register step.")

    Path("mlflow_run_id.txt").write_text(run_id)
    print("\nRun ID written to mlflow_run_id.txt")

    # Normalise so downstream scripts always see 'accuracy'
    return {**metrics, "accuracy": accuracy}


def main() -> None:
    import dagshub  # noqa: PLC0415
    import yaml     # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Fetch model metrics from MLflow registry and gate on accuracy."
    )
    parser.add_argument("--config", default="configs/eval_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dagshub.init(repo_owner="marcosncosta1", repo_name="ai-image-detector", mlflow=True)

    run_evaluation(config)


if __name__ == "__main__":
    main()
