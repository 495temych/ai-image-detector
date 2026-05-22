from src.evaluate import compute_metrics, parse_pipeline_output


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


def test_parse_pipeline_output_real():
    raw = [{"label": "REAL", "score": 0.98}, {"label": "FAKE", "score": 0.02}]
    label, score = parse_pipeline_output(raw, fake_label="FAKE")
    assert label == 0        # REAL → not fake
    assert abs(score - 0.02) < 1e-6  # fake probability


def test_parse_pipeline_output_fake():
    raw = [{"label": "FAKE", "score": 0.91}, {"label": "REAL", "score": 0.09}]
    label, score = parse_pipeline_output(raw, fake_label="FAKE")
    assert label == 1        # FAKE → is fake
    assert abs(score - 0.91) < 1e-6
