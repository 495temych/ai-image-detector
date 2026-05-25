"""Shared MLOps utilities used by evaluate.py and register.py."""

# Metric key aliases — checked in priority order.
# test_acc beats val_acc so we prefer the held-out test set when both exist.
ACCURACY_ALIASES = [
    "accuracy",
    "test_acc",
    "val_acc",
    "test_accuracy",
    "val_accuracy",
    "acc",
]


def extract_accuracy(metrics: dict) -> float | None:
    """Return the accuracy value from a metrics dict, trying known aliases.

    Returns None if no accuracy-like key is found.
    """
    for key in ACCURACY_ALIASES:
        if key in metrics:
            return float(metrics[key])
    return None
