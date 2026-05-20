"""Model loading and inference — ONNX for /predict, PyTorch for /explain."""
import base64
import io
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

MODEL_DIR = Path(__file__).parent
ONNX_PATH = MODEL_DIR / "model.onnx"
PT_PATH = MODEL_DIR / "best_weights.pt"

MODEL_VERSION = "efficientnet_v1"
RUN_ID = os.getenv("MLFLOW_RUN_ID", "2561d86a3f22495e91b2cc7d3d1d3497")

CLASSES = {0: "fake", 1: "real"}

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_onnx() -> ort.InferenceSession:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at {ONNX_PATH}. Run download_artifacts.py first.")
    return ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])


def _load_pytorch() -> torch.nn.Module:
    if not PT_PATH.exists():
        raise FileNotFoundError(f"PyTorch weights not found at {PT_PATH}. Run download_artifacts.py first.")
    net = models.efficientnet_b0(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 2),
    )
    state = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    net.load_state_dict(state)
    net.eval()
    return net


# Load both models once at import time (startup).
onnx_session: ort.InferenceSession = _load_onnx()
pytorch_model: torch.nn.Module = _load_pytorch()

# GradCAM serialisation lock — pytorch_model is a shared singleton; concurrent
# forward+backward passes race on the hook lists and corrupt each other's maps.
_gradcam_lock = threading.Lock()


def _preprocess(image_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _TRANSFORM(pil).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor.numpy(), pil


def predict_onnx(image_bytes: bytes) -> tuple[str, float]:
    """Run ONNX inference; returns (label, confidence)."""
    inp, _ = _preprocess(image_bytes)
    input_name = onnx_session.get_inputs()[0].name
    logits = onnx_session.run(None, {input_name: inp})[0]  # (1, 2)
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()[0]
    class_idx = int(np.argmax(probs))
    return CLASSES[class_idx], float(probs[class_idx])


def explain_gradcam(image_bytes: bytes) -> tuple[str, float, str]:
    """Run GradCAM on PyTorch model; returns (label, confidence, gradcam_base64)."""
    tensor, pil_orig = _preprocess(image_bytes)
    x = torch.tensor(tensor, requires_grad=False)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    target_layer = pytorch_model.features[-1]

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    # Serialise forward+backward: concurrent requests share pytorch_model and
    # would stack their hooks on the same layer, corrupting each other's maps.
    with _gradcam_lock:
        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        try:
            logits = pytorch_model(x)
            probs = F.softmax(logits, dim=1)
            class_idx = int(probs.argmax())
            confidence = float(probs[0, class_idx])

            pytorch_model.zero_grad()
            logits[0, class_idx].backward()
        finally:
            h_fwd.remove()
            h_bwd.remove()

    act = activations[0].squeeze(0)       # (C, H, W)
    grad = gradients[0].squeeze(0)        # (C, H, W)
    weights = grad.mean(dim=(1, 2))       # (C,)
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))  # (H, W)

    cam = cam.numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig = np.array(pil_orig.resize((224, 224))).astype(np.float32)
    overlay = (0.5 * orig + 0.5 * heatmap).clip(0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode()

    return CLASSES[class_idx], confidence, gradcam_b64
