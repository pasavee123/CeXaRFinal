from __future__ import annotations

from typing import Tuple
import numpy as np
from PIL import Image
import cv2


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Tuple[Image.Image, Image.Image]:
    """Create colorized heatmap and overlay on original image.

    Returns (heatmap_img, overlay_img) as PIL Images.
    """
    rgb = ensure_rgb(image)
    img_np = np.array(rgb)
    h, w = img_np.shape[:2]

    # Resize heatmap to match image size
    if heatmap.shape[0] != h or heatmap.shape[1] != w:
        heatmap_resized = cv2.resize(heatmap, (w, h))
    else:
        heatmap_resized = heatmap

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (alpha * colored_rgb + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(colored_rgb), Image.fromarray(overlay)


