from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAM
except Exception:  # pragma: no cover
    GradCAM = None


def run_gradcam(model: nn.Module, target_layers, input_tensor: torch.Tensor) -> np.ndarray:
    if GradCAM is None:
        raise RuntimeError("pytorch-grad-cam not available")
    with GradCAM(model=model, target_layers=target_layers, use_cuda=(next(model.parameters()).device.type == "cuda")) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    heatmap = grayscale_cam[0]
    return heatmap


