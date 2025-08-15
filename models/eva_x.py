from __future__ import annotations

import os
import sys
import logging
import importlib.util
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
import timm
import types

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


def _register_eva_x_models() -> bool:
    """Register EVA-X models with timm registry by importing models_eva.py.

    Returns True if successful, False otherwise.
    """
    repo_root = Path(__file__).resolve().parents[1]  # project root
    eva_cls_path = repo_root / "EVA-X[repo]" / "classification" / "models" / "models_eva.py"
    rope_path = repo_root / "EVA-X[repo]" / "classification" / "models" / "rope.py"
    
    logger.info("Checking EVA-X paths:")
    logger.info("  Repo root: %s", repo_root)
    logger.info("  models_eva.py: %s (exists: %s)", eva_cls_path, eva_cls_path.exists())
    logger.info("  rope.py: %s (exists: %s)", rope_path, rope_path.exists())
    
    if not eva_cls_path.exists():
        logger.error("EVA-X reference model file not found at %s", str(eva_cls_path))
        logger.error("Please ensure EVA-X[repo] directory structure is intact")
        return False
    
    if not rope_path.exists():
        logger.error("EVA-X rope module not found at %s", str(rope_path))
        return False

    try:
        # Ensure stubs are available before importing EVA-X
        _ensure_dependency_stubs()
        
        # Import EVA-X models using direct module loading
        eva_models_dir = str(eva_cls_path.parent)
        eva_classification_dir = str(eva_cls_path.parent.parent)
        
        # Add necessary paths to sys.path
        original_path = sys.path.copy()
        sys.path.insert(0, eva_models_dir)
        sys.path.insert(0, eva_classification_dir)
        
        try:
            # Read and execute rope module first
            rope_code = rope_path.read_text(encoding='utf-8')
            rope_module = types.ModuleType("rope")
            exec(rope_code, rope_module.__dict__)
            sys.modules["rope"] = rope_module
            logger.info("Successfully loaded rope module")
            
            # Read and modify models_eva to handle imports
            eva_code = eva_cls_path.read_text(encoding='utf-8')
            
            # Replace relative imports
            eva_code = eva_code.replace("from .rope import *", "from rope import *")
            
            # Create module and execute
            eva_module = types.ModuleType("models_eva_custom")
            eva_module.__file__ = str(eva_cls_path)
            eva_module.__dict__['rope'] = rope_module
            
            # Copy rope module contents to eva_module namespace
            for name in dir(rope_module):
                if not name.startswith('_'):
                    eva_module.__dict__[name] = getattr(rope_module, name)
            
            # Execute the modified code
            exec(eva_code, eva_module.__dict__)
            sys.modules["models_eva_custom"] = eva_module
            
            logger.info("Successfully registered EVA-X models with timm registry")
            return True
            
        finally:
            # Restore original sys.path
            sys.path[:] = original_path
            
    except Exception as e:
        logger.error("Failed to import EVA-X models: %s", e)
        import traceback
        logger.error("Full traceback: %s", traceback.format_exc())
        return False


def _get_model_name_from_path(model_path: str) -> str:
    """Extract model architecture name from model path or use default."""
    # Default to small model
    default_model = "eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE"
    
    # Try to infer from path
    if "tiny" in model_path.lower():
        return "eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE"
    elif "base" in model_path.lower():
        return "eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE"
    elif "small" in model_path.lower():
        return "eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE"
    
    return default_model


# Removed fallback classifier - only use real EVA-X models


@lru_cache(maxsize=2)
def create_model(model_name: str, device: str = "cuda") -> Tuple[nn.Module, Callable[[Image.Image], torch.Tensor], List[str]]:
    """Create EVA-X model using timm registry - the correct EVA-X approach."""
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # Setup dependency stubs
    _ensure_dependency_stubs()
    
    # Register EVA-X models with timm - REQUIRED
    eva_registration_success = _register_eva_x_models()
    if not eva_registration_success:
        raise RuntimeError(
            "EVA-X model registration failed. This medical application requires authentic EVA-X models. "
            "Please ensure EVA-X[repo]/classification/models/models_eva.py exists and all dependencies are installed."
        )
    
    # Download and verify checkpoint
    ckpt_path = _download_checkpoint(model_name)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise RuntimeError(
            f"Model checkpoint not found: {model_name}. "
            "Medical diagnosis requires trained weights. Please provide valid model path or HuggingFace Hub ID."
        )
    
    # Load checkpoint to determine architecture and classes
        state = torch.load(ckpt_path, map_location=device_t)
        if isinstance(state, dict):
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            elif "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
    
    # Determine number of classes from checkpoint
        num_classes_from_ckpt: Optional[int] = None
        if isinstance(state, dict):
            hw = state.get("head.weight")
            hb = state.get("head.bias")
            if isinstance(hw, torch.Tensor):
                num_classes_from_ckpt = int(hw.shape[0])
            elif isinstance(hb, torch.Tensor):
                num_classes_from_ckpt = int(hb.shape[0])
    
    if num_classes_from_ckpt is None:
        raise RuntimeError("Cannot determine number of classes from checkpoint. Invalid model file.")
    
    # Determine model architecture from path
    model_arch = _get_model_name_from_path(model_name)
    
    # Create model using timm - MUST be authentic EVA-X
    try:
        model = timm.create_model(
            model_arch,
            pretrained=False,
            num_classes=num_classes_from_ckpt,
            img_size=224,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
        )
        logger.info("Successfully created authentic EVA-X model (%s) with %d classes", model_arch, num_classes_from_ckpt)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create EVA-X model ({model_arch}): {e}. "
            f"This medical application requires authentic EVA-X models only. "
            f"Please ensure EVA-X[repo] is properly installed and all dependencies are available."
        )
    
    # Load weights
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("Loaded EVA-X weights: missing=%s unexpected=%s", missing, unexpected)
        if missing:
            logger.warning("Missing keys in checkpoint: %s", missing)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    # Setup label mapping
    label_map = _get_label_map_by_count(num_classes_from_ckpt)

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
    """Generate GradCAM heatmap for Vision Transformer models."""
    device = next(model.parameters()).device
    
    # For Vision Transformer, use attention rollout or gradient-based methods
    if GradCAM is None:
        # Fallback: simple gradient-based saliency
        image_tensor = image_tensor.requires_grad_(True)
        scores = model(image_tensor)
        # Use highest confidence prediction for grad computation
        if scores.dim() > 1:
            top_idx = scores.sigmoid().argmax(dim=1)
            scores = scores.gather(1, top_idx.unsqueeze(1)).squeeze()
        scores.backward()
        grads = image_tensor.grad.detach().abs().mean(dim=1, keepdim=True)
        heatmap = grads[0, 0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        return heatmap.cpu().numpy()

    # For EVA-X Vision Transformer, target the last attention block's norm layer
    target_layers = []
    
    # Try to find appropriate layers for ViT
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        # Use the last transformer block's norm layer
        last_block = model.blocks[-1]
        if hasattr(last_block, 'norm1'):
            target_layers = [last_block.norm1]
        elif hasattr(last_block, 'ln_1'):
            target_layers = [last_block.ln_1]
        else:
            # Fallback to the entire last block
            target_layers = [last_block]
    
    # If no suitable layer found, try patch embedding
    if not target_layers and hasattr(model, 'patch_embed'):
        if hasattr(model.patch_embed, 'proj'):
            target_layers = [model.patch_embed.proj]
    
    # Last resort: use model head
    if not target_layers and hasattr(model, 'head'):
        target_layers = [model.head]
    
    if not target_layers:
        # Ultimate fallback
        target_layers = [list(model.children())[-1]]
    
    try:
    with GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type == "cuda")) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=None)
        heatmap = grayscale_cam[0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        return heatmap
    except Exception as e:
        logger.warning("GradCAM failed with error: %s. Using gradient saliency fallback.", e)
        # Fallback to simple gradient method
        image_tensor = image_tensor.requires_grad_(True)
        scores = model(image_tensor)
        if scores.dim() > 1:
            top_idx = scores.sigmoid().argmax(dim=1)
            scores = scores.gather(1, top_idx.unsqueeze(1)).squeeze()
        scores.backward()
        grads = image_tensor.grad.detach().abs().mean(dim=1, keepdim=True)
        heatmap = grads[0, 0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        return heatmap.cpu().numpy()


