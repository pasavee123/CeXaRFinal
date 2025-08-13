from __future__ import annotations

import os
import sys
import logging
import importlib.util
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

try:
    from pytorch_grad_cam import GradCAM
except Exception:  # pragma: no cover
    GradCAM = None

logger = logging.getLogger(__name__)


def _ensure_dependency_stubs() -> None:
    """Provide lightweight stubs for optional EVA-X deps when unavailable.

    - apex.normalization.FusedLayerNorm -> torch.nn.LayerNorm
    - xformers.ops.memory_efficient_attention -> naive attention fallback
    """
    # Stub apex.normalization.FusedLayerNorm
    try:
        import apex  # type: ignore
        from apex.normalization import FusedLayerNorm as _FLN  # noqa: F401
    except Exception:
        import types
        apex_mod = types.ModuleType("apex")
        normalization_mod = types.ModuleType("apex.normalization")

        class FusedLayerNorm(nn.LayerNorm):
            pass

        normalization_mod.FusedLayerNorm = FusedLayerNorm
        apex_mod.normalization = normalization_mod  # type: ignore[attr-defined]
        sys.modules.setdefault("apex", apex_mod)
        sys.modules.setdefault("apex.normalization", normalization_mod)

    # Stub xformers.ops.memory_efficient_attention
    try:
        import xformers.ops as _xops  # type: ignore # noqa: F401
    except Exception:
        import types
        xops_mod = types.ModuleType("xformers.ops")

        def memory_efficient_attention(q, k, v):
            # q,k,v shapes: [B, N, H, C] or [B, H, N, C] depending on caller; EVA uses [B, N, num_heads, C]
            if q.dim() == 4 and q.shape[1] != q.shape[2]:
                # assume [B, N, H, C]
                B, N, H, C = q.shape
                q_ = q.permute(0, 2, 1, 3).reshape(B * H, N, C)
                k_ = k.permute(0, 2, 1, 3).reshape(B * H, N, C)
                v_ = v.permute(0, 2, 1, 3).reshape(B * H, N, C)
                attn = torch.softmax((q_ @ k_.transpose(-2, -1)) / (C ** 0.5), dim=-1)
                out = attn @ v_
                out = out.reshape(B, H, N, C).permute(0, 2, 1, 3).contiguous()
                out = out.view(B, N, H * C)
                return out
            else:
                # assume [B, H, N, C]
                B, H, N, C = q.shape
                q_ = q.reshape(B * H, N, C)
                k_ = k.reshape(B * H, N, C)
                v_ = v.reshape(B * H, N, C)
                attn = torch.softmax((q_ @ k_.transpose(-2, -1)) / (C ** 0.5), dim=-1)
                out = attn @ v_
                out = out.reshape(B, H, N, C)
                out = out.permute(0, 2, 1, 3).contiguous().view(B, N, H * C)
                return out

        xops_mod.memory_efficient_attention = memory_efficient_attention
        sys.modules.setdefault("xformers.ops", xops_mod)


def _get_preprocess_fn(model_name: str, input_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    mean = (0.49185243, 0.49185243, 0.49185243)
    std = (0.28509309, 0.28509309, 0.28509309)
    transform = transforms.Compose([
        transforms.Resize(int(256), interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform


def _get_label_map_default16() -> List[str]:
    # 16-class extension for demo
    return [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia", "Fracture", "Lesion"
    ]


def _get_label_map_chexpert14() -> List[str]:
    # Common CheXpert/ChestX-ray14 label set (14 classes)
    return [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]


def _get_label_map_by_count(num_classes: int) -> List[str]:
    if num_classes == 14:
        return _get_label_map_chexpert14()
    if num_classes == 16:
        return _get_label_map_default16()
    # Fallback: create generic labels
    return [f"Class_{i}" for i in range(num_classes)]


def _download_checkpoint(model_name: str) -> Optional[str]:
    if model_name in (None, "", "local"):
        return None
    if os.path.isfile(model_name):
        return model_name
    # Support explicit file inside HF repo: "repo_id:filename"
    if ":" in model_name:
        repo_id, filename = model_name.split(":", 1)
        logger.info("Downloading specific file from hub: %s (%s)", repo_id, filename)
        return hf_hub_download(repo_id=repo_id, filename=filename)
    # Otherwise, try common filenames
    try:
        return hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    except Exception:
        try:
            return hf_hub_download(repo_id=model_name, filename="model.pth")
        except Exception as e:
            logger.warning("Failed to download weights from hub: %s", e)
            return None


def _import_eva_x_factory() -> Optional[Callable[..., nn.Module]]:
    """Dynamically import EVA-X classification model factory from reference repo.

    Returns a callable like `eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE` if available,
    else None.
    """
    repo_root = Path(__file__).resolve().parents[1]  # project root
    eva_cls_path = repo_root / "EVA-X[repo]" / "classification" / "models" / "models_eva.py"
    if not eva_cls_path.exists():
        logger.warning("EVA-X reference model file not found at %s", str(eva_cls_path))
        return None

    try:
        spec = importlib.util.spec_from_file_location("eva_models_local", str(eva_cls_path))
        if spec is None or spec.loader is None:
            logger.warning("Failed to load EVA-X spec from %s", str(eva_cls_path))
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        # Prefer small patch16 architecture used by provided checkpoints
        factory_name = "eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE"
        if hasattr(module, factory_name):
            return getattr(module, factory_name)
        # Fallbacks
        for name in (
            "eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE",
            "eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE",
        ):
            if hasattr(module, name):
                return getattr(module, name)
        logger.warning("No suitable EVA-X factory found in module %s", eva_cls_path.name)
        return None
    except Exception as e:
        logger.warning("Failed to import EVA-X model due to: %s", e)
        return None


class _FallbackClassifier(nn.Module):
    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def create_model(model_name: str, device: str = "cuda") -> Tuple[nn.Module, Callable[[Image.Image], torch.Tensor], List[str]]:
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    label_map = _get_label_map_default16()

    # Try to build real EVA-X model
    model: nn.Module
    _ensure_dependency_stubs()
    factory = _import_eva_x_factory()
    if factory is not None:
        try:
            # Instantiate with neutral head; we'll reset to checkpoint classes after reading weights
            model = factory(pretrained=False, num_classes=0)
            logger.info("Initialized real EVA-X model via %s", factory.__name__)
        except Exception as e:
            logger.warning("EVA-X factory instantiation failed (%s). Falling back to small CNN.", e)
            model = _FallbackClassifier(num_classes=len(label_map))
    else:
        model = _FallbackClassifier(num_classes=len(label_map))
        logger.warning("Using fallback classifier as EVA-X import was unavailable.")

    # Load checkpoint
    ckpt_path = _download_checkpoint(model_name)
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device_t)
        # Common containers: {"model": ...}, {"state_dict": ...}
        if isinstance(state, dict):
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            elif "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
        # Try inferring classifier size from checkpoint and reset head accordingly (for EVA-X VisionTransformer)
        num_classes_from_ckpt: Optional[int] = None
        if isinstance(state, dict):
            hw = state.get("head.weight")
            hb = state.get("head.bias")
            if isinstance(hw, torch.Tensor):
                num_classes_from_ckpt = int(hw.shape[0])
            elif isinstance(hb, torch.Tensor):
                num_classes_from_ckpt = int(hb.shape[0])
        if num_classes_from_ckpt is not None and hasattr(model, "reset_classifier"):
            try:
                model.reset_classifier(num_classes_from_ckpt)
                label_map = _get_label_map_by_count(num_classes_from_ckpt)
                logger.info("Reset classifier to %d classes inferred from checkpoint.", num_classes_from_ckpt)
            except Exception:
                logger.warning("Failed to reset classifier to %d; continuing.", num_classes_from_ckpt)
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info("Loaded EVA-X weights: missing=%s unexpected=%s", missing, unexpected)
        except Exception as e:
            logger.warning("Failed to load EVA-X checkpoint (%s). Using randomly initialized weights.", e)
    else:
        logger.info("No checkpoint provided; using randomly initialized weights.")

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

    # Use current label_map length from classifier if it matches known sets
    # We cannot access label_map here directly; rebuild based on classifier head if possible
    num_classes = logits.shape[-1]
    if num_classes == 14:
        label_map = _get_label_map_chexpert14()
    elif num_classes == 16:
        label_map = _get_label_map_default16()
    else:
        label_map = [f"Class_{i}" for i in range(num_classes)]
    results: List[Dict[str, float]] = []
    for label, p, l in zip(label_map, probs_np.tolist(), logits_np.tolist()):
        results.append({"label": label, "confidence": float(p), "logit": float(l)})
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def get_gradcam_heatmap(model: nn.Module, target_layer, image_tensor: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    # For ViT-based EVA-X, GradCAM conv target may not exist; fallback to simple input-grad saliency
    if GradCAM is None:
        image_tensor = image_tensor.requires_grad_(True)
        scores = model(image_tensor)
        top_idx = scores.sigmoid().mean(dim=1).argmax()
        scores[:, top_idx].backward()
        grads = image_tensor.grad.detach().abs().mean(dim=1, keepdim=True)
        heatmap = grads[0, 0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        return heatmap.cpu().numpy()

    # Try to find a conv layer; if not found, use last module
    last_conv = None
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            last_conv = m
            break
    target_layers = [last_conv] if last_conv is not None else [list(model.children())[-1]]

    with GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type == "cuda")) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=None)
        heatmap = grayscale_cam[0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        return heatmap


