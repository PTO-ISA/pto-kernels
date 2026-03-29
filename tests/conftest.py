from pathlib import Path
import os
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

try:
    import torch
except ImportError:
    torch = None


NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
if torch is not None and hasattr(torch, "npu"):
    torch.npu.config.allow_internal_format = False
    torch.npu.set_device(NPU_DEVICE)


@pytest.fixture(scope="session")
def npu_device():
    return NPU_DEVICE
