import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# conftest.py injects a stub api.model before this import runs,
# so no real ONNX / PyTorch weights are loaded.
from api.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_jpeg_bytes(color: tuple = (100, 150, 200)) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    """TestClient backed by the stub model (no real inference)."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_predict_returns_valid_structure(client):
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ("real", "fake")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_real_image(client):
    # Stub returns ("real", 0.9) → label must be "real"
    r = client.post(
        "/predict",
        files={"file": ("img.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert r.json()["label"] == "real"


def test_predict_rejects_non_image(client):
    r = client.post(
        "/predict",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert r.status_code == 415
