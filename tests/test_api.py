import io
import json
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
    """Patch the _get_* lazy helpers so no real model files are needed."""
    mock_session = mock.MagicMock()
    mock_session.run.return_value = [np.array([[0.1, 0.9]])]  # logits: REAL (idx 1) wins

    mock_fe = mock.MagicMock()
    mock_fe.return_value = {
        "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)
    }

    with (
        mock.patch("api.main._get_session", return_value=mock_session),
        mock.patch("api.main._get_feature_extractor", return_value=mock_fe),
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
