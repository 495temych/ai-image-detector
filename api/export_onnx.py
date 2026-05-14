"""Convert best_weights.pt → model.onnx (single file, weights embedded)."""
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

API_DIR = Path(__file__).parent
PT_PATH = API_DIR / "best_weights.pt"
ONNX_PATH = API_DIR / "model.onnx"

net = models.efficientnet_b0(weights=None)
in_features = net.classifier[1].in_features
net.classifier[1] = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features, 2),
)
state = torch.load(PT_PATH, map_location="cpu", weights_only=False)
net.load_state_dict(state)
net.eval()

dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(
    net,
    dummy,
    str(ONNX_PATH),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
print(f"Exported → {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
