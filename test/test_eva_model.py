import os
from PIL import Image
import numpy as np
import torch

from models.eva_x import create_model, analyze_image, get_gradcam_heatmap


def test_create_model_cpu():
    model, preprocess, label_map = create_model(model_name="local", device="cpu")
    assert hasattr(model, 'forward')
    img = Image.new('RGB', (256, 256), color=(128, 128, 128))
    tensor = preprocess(img)
    assert tensor.shape[1:] == (224, 224)
    assert len(label_map) >= 14


def test_analyze_image_output_fields():
    model, preprocess, label_map = create_model(model_name="local", device="cpu")
    img = Image.new('RGB', (256, 256), color=(128, 128, 128))
    results = analyze_image(model, preprocess, img)
    assert isinstance(results, list)
    assert all({'label', 'confidence', 'logit'} <= set(r.keys()) for r in results)


def test_gradcam_shape():
    model, preprocess, label_map = create_model(model_name="local", device="cpu")
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    tensor = preprocess(img).unsqueeze(0)
    heatmap = get_gradcam_heatmap(model, None, tensor)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)


