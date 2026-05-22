"""
Shared pytest configuration.

api.model is injected into sys.modules here — BEFORE any test file is
collected — so pytest never loads the real ONNX / PyTorch weights.
The injection only happens if api.model hasn't already been imported
(e.g. if you run tests after importing the real API in the same process).
"""
import sys
from types import ModuleType
from unittest import mock

_MODULE_NAME = "api.model"

if _MODULE_NAME not in sys.modules:
    _stub = ModuleType(_MODULE_NAME)

    # Scalar constants that api.main imports directly
    _stub.MODEL_VERSION = "efficientnet_v1"
    _stub.RUN_ID = "2561d86a3f22495e91b2cc7d3d1d3497"

    # Inference helpers — return deterministic values so tests are fast & stable
    _stub.predict_onnx = mock.MagicMock(return_value=("real", 0.9))
    _stub.explain_gradcam = mock.MagicMock(
        return_value=("real", 0.9, "stub_gradcam_base64==")
    )

    sys.modules[_MODULE_NAME] = _stub
