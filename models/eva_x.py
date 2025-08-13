from __future__ import annotations

import os
import logging
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from huggingface_hub import snapshot_download, hf_hub_download

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:  # pragma: no cover
    GradCAM = None

logger = logging.getLogger(__name__)


# Minimal EVA-X like ViT head for demo; accepts external weights.
class SimpleViTClassifier(nn.Module):
    def __init__(self, num_classes: int = 16, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _get_preprocess_fn(model_name: str, input_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    # From EVA-X classification/utils/datasets.py build_transform mean/std for 'eva' models
    # mean=(0.49185243, 0.49185243, 0.49185243), std=(0.28509309, 0.28509309, 0.28509309)
    mean = (0.49185243, 0.49185243, 0.49185243)
    std = (0.28509309, 0.28509309, 0.28509309)

    transform = transforms.Compose([
        transforms.Resize(int(256), interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform


def _get_label_map() -> List[str]:
    # 16-class mapping seen across EVA-X configs (interpolate14to16 implies 16). Provide common ChestX-ray14 labels + 2 extras.
    return [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia", "Fracture", "Lesion"
    ]


def _load_weights_into_model(model: nn.Module, model_name: str, map_location: Optional[torch.device] = None) -> None:
    if model_name in (None, "", "local"):
        logger.info("Using randomly initialized demo weights.")
        return

    ckpt_path: Optional[str] = None
    if os.path.isfile(model_name):
        ckpt_path = model_name
    else:
        try:
            ckpt_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        except Exception:
            try:
                ckpt_path = hf_hub_download(repo_id=model_name, filename="model.pth")
            except Exception as e:
                logger.warning("Failed to download weights from hub: %s", e)

    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=map_location or "cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("Loaded weights: missing=%s unexpected=%s", missing, unexpected)
    else:
        logger.info("Weights not provided; using random init.")


def create_model(model_name: str, device: str = "cuda") -> Tuple[nn.Module, Callable[[Image.Image], torch.Tensor], List[str]]:
    """Create model, preprocessing, and label map.

    - Supports local weight path or HF Hub ID via model_name.
    - Uses EVA-X preprocessing stats.
    """
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    label_map = _get_label_map()
    model = SimpleViTClassifier(num_classes=len(label_map), img_size=224)

    # New convention: "repo_id:filename" targets a specific file inside a HF repo
    if model_name not in (None, "", "local") and ":" in model_name and not os.path.isfile(model_name):
        try:
            repo_id, filename = model_name.split(":", 1)
            logger.info("Downloading specific file from hub: %s (%s)", repo_id, filename)
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
            state = torch.load(ckpt_path, map_location=device_t)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info("Loaded weights (explicit file): missing=%s unexpected=%s", missing, unexpected)
        except Exception as e:
            logger.warning("Explicit HF file download failed (%s). Falling back to default loader.", e)
            _load_weights_into_model(model, model_name, map_location=device_t)
    else:
        _load_weights_into_model(model, model_name, map_location=device_t)
    model.eval().to(device_t)

    preprocess_fn = _get_preprocess_fn(model_name, input_size=224)
    return model, preprocess_fn, label_map


@torch.no_grad()
def analyze_image(model: nn.Module, preprocess_fn: Callable[[Image.Image], torch.Tensor], image: Image.Image) -> List[Dict[str, float]]:
    tensor = preprocess_fn(image).unsqueeze(0).to(next(model.parameters()).device)
    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0)

    probs_np = probs.detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()[0]

    label_map = _get_label_map()
    results: List[Dict[str, float]] = []
    for label, p, l in zip(label_map, probs_np.tolist(), logits_np.tolist()):
        results.append({"label": label, "confidence": float(p), "logit": float(l)})
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def get_gradcam_heatmap(model: nn.Module, target_layer, image_tensor: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    if GradCAM is None:
        # Fallback simple saliency via absolute gradients norm over input
        image_tensor = image_tensor.requires_grad_(True)
        scores = model(image_tensor)
        top_idx = scores.sigmoid().mean(dim=1).argmax()
        scores[:, top_idx].backward()
        grads = image_tensor.grad.detach().abs().mean(dim=1, keepdim=True)  # Bx1xHxW
        heatmap = grads[0, 0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        return heatmap.cpu().numpy()

    # Try to pick a convolutional layer inside model.features for CAM
    last_conv = None
    if hasattr(model, "features"):
        for m in reversed(model.features):
            if isinstance(m, nn.Conv2d):
                last_conv = m
                break
    target_layers = [last_conv] if last_conv is not None else [list(model.children())[-1]]

    with GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type == "cuda")) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=None)
        heatmap = grayscale_cam[0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        return heatmap


