import os
import io
import time
import logging
from typing import Tuple, List, Dict, Any

import gradio as gr
from PIL import Image
import numpy as np
import torch

from dotenv import load_dotenv

from models.eva_x import create_model, analyze_image, get_gradcam_heatmap
from llm.openai_client import generate_explanation, chat_reply_stream
from utils.image import ensure_rgb, overlay_heatmap_on_image
from utils.security import RateLimiter, validate_image_file


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CeXaR")


MODEL_NAME = os.getenv(
    "CEXAR_MODEL",
    "MapleF/eva_x:eva_x_small_patch16_merged520k_mim_chexpert_ft.pth",
)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

rate_limiter = RateLimiter(max_requests=10, per_seconds=60)


# Model is now cached automatically via @lru_cache decorator in create_model()
def _get_model_info(model_name: str, device_str: str):
    """Get model info with caching. Model creation is cached automatically."""
    device = device_str if device_str in ("cpu", "cuda") else DEFAULT_DEVICE
    return model_name, device


def process(image: Image.Image, model_name: str, device_choice: str, alpha: float) -> Tuple[Image.Image, Image.Image, List[Dict[str, Any]], str, Dict[str, Any]]:
    start_time = time.time()

    # Rate limit
    rate_limiter.check()

    # Basic validation (no need to re-encode for validation)
    img_rgb = ensure_rgb(image)
    # Simple size check without re-encoding
    if img_rgb.width * img_rgb.height > 10000 * 10000:  # ~100MP limit
        raise ValueError("Image too large. Please use images smaller than 100 megapixels.")

    # Get model info and create/retrieve cached model
    actual_model_name, device = _get_model_info(model_name, device_choice)
    
    try:
        # This call is cached - won't reload if same parameters
        model, preprocess_fn, label_map = create_model(model_name=actual_model_name, device=device)
        logger.info("Using model: %s on device: %s", actual_model_name, device)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise RuntimeError(f"Model loading failed: {e}. Please check model path and ensure you have valid trained weights.")

    # Inference
    try:
        diagnoses = analyze_image(model, preprocess_fn, img_rgb)
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise RuntimeError(f"Medical image analysis failed: {e}")

    # Grad-CAM with improved error handling
    try:
        image_tensor = preprocess_fn(img_rgb).unsqueeze(0).to(device)
        heatmap = get_gradcam_heatmap(model, None, image_tensor)  # target_layer auto-detected
        heatmap_img, overlay_img = overlay_heatmap_on_image(img_rgb, heatmap, alpha=alpha)
    except Exception as e:
        logger.warning("Grad-CAM generation failed: %s. Continuing without heatmap.", e)
        heatmap_img = img_rgb.copy()
        overlay_img = img_rgb.copy()

    # LLM explanation
    image_meta = {"width": img_rgb.width, "height": img_rgb.height}
    try:
        explanation = generate_explanation(diagnoses, image_meta)
    except Exception as e:
        logger.warning("LLM explanation failed: %s", e)
        explanation = "ระบบอธิบายผลไม่พร้อมใช้งานในขณะนี้ กรุณาพิจารณาผลการวิเคราะห์จากโมเดลโดยตรง"

    latency = time.time() - start_time
    meta = {
        "model": actual_model_name,
        "device": str(device),
        "latency_sec": round(latency, 3),
        "num_classes": len(label_map)
    }

    table_rows = [[d["label"], round(float(d["confidence"]), 4), round(float(d["logit"]), 4)] for d in diagnoses]
    return img_rgb, overlay_img, table_rows, explanation, meta


def build_interface():
    base_css = """
    body { background: linear-gradient(180deg,#eaf5ff 0%, #ffffff 60%); }
    .c-card { background: #fff; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); padding: 18px; }
    .c-header { margin-bottom: 8px; }
    .c-sub { color: #546e7a; margin-bottom: 16px; }
    """

    with gr.Blocks(title="CeXaR: Chest X-ray Analyzer", css=base_css) as demo:
        chat_state = gr.State({"history": [], "context": {}})

        with gr.Row(elem_classes=["c-header"]):
            gr.Markdown("### AI-Powered Chest X-ray Analysis")
        gr.Markdown("Combining EVA-X diagnostics with explainable AI for transparent medical decision-making", elem_classes=["c-sub"])

        with gr.Row():
            with gr.Column(scale=6, elem_classes=["c-card"]):
                gr.Markdown("**Upload X-ray Image**")
                image_in = gr.Image(type="pil", label=None, height=360)
                analyze_btn = gr.Button("Analyze", variant="primary")

                img_out = gr.Image(type="pil", label="Original")
                overlay_out = gr.Image(type="pil", label="Grad-CAM Overlay", show_download_button=True)
                table_out = gr.Dataframe(headers=["label", "confidence", "logit"], label="Predictions", interactive=False)
                explanation_out = gr.Textbox(label="AI Explanation (TH)", lines=6)
                meta_out = gr.JSON(label="Meta")

            with gr.Column(scale=6):
                with gr.Column(elem_classes=["c-card"]):
                    gr.Markdown("**AI Explanation**")

                with gr.Column(elem_classes=["c-card"]):
                    gr.Markdown("**Ask CeXaR**")
                    chatbot = gr.Chatbot(label=None, height=340)
                    chat_in = gr.Textbox(label=None, placeholder="Ask about the findings...", lines=2)
                    with gr.Row():
                        send_chat = gr.Button("Send")

        def process_with_ctx(image):
            try:
                img_rgb, overlay_img, rows, explanation, meta = process(
                    image, MODEL_NAME, DEFAULT_DEVICE, 0.45
                )
                ctx = {"diagnoses": [{"label": r[0], "confidence": float(r[1]), "logit": float(r[2])} for r in rows], "meta": meta}
                # Initialize chat with assistant's initial explanation and store context
                initial_history = [{"role": "assistant", "content": explanation}]
                initial_pairs = [("", explanation)]
                state = {"history": initial_history, "context": ctx}
                return img_rgb, overlay_img, rows, explanation, meta, state, initial_pairs
            except Exception as e:
                logger.error("Processing failed: %s", e)
                error_msg = f"การวิเคราะห์ภาพล้มเหลว: {str(e)}"
                # Return error state
                error_image = Image.new('RGB', (224, 224), color=(200, 200, 200))
                error_rows = [["Error", 0.0, 0.0]]
                error_meta = {"error": str(e)}
                error_state = {"history": [], "context": {}}
                error_pairs = []
                return error_image, error_image, error_rows, error_msg, error_meta, error_state, error_pairs

        analyze_btn.click(
            fn=process_with_ctx,
            inputs=[image_in],
            outputs=[img_out, overlay_out, table_out, explanation_out, meta_out, chat_state, chatbot],
        )

        def _pairs(history):
            pairs, u = [], None
            for m in history:
                if m["role"] == "user":
                    u = m["content"]
                elif m["role"] == "assistant" and u is not None:
                    pairs.append((u, m["content"]))
                    u = None
            return pairs

        def on_send(user_msg, state):
            state = state or {"history": [], "context": {}}
            context = state.get("context", {})
            prior_pairs = _pairs(state.get("history", []))
            msgs = state.get("history", []) + [{"role": "user", "content": user_msg}]

            assistant_so_far = ""
            try:
                for chunk in chat_reply_stream(msgs, context):
                    assistant_so_far += chunk
                    running_pairs = prior_pairs + [(user_msg, assistant_so_far)]
                    running_state = {
                        "history": msgs + [{"role": "assistant", "content": assistant_so_far}],
                        "context": context,
                    }
                    yield running_pairs, running_state, ""
            except Exception:
                logger.exception("Streaming chat failed; falling back to single response state update.")
                new_history = msgs + [{"role": "assistant", "content": "เกิดข้อผิดพลาดในการสตรีมคำตอบ"}]
                state["history"] = new_history
                yield _pairs(new_history), state, ""

        send_chat.click(on_send, inputs=[chat_in, chat_state], outputs=[chatbot, chat_state, chat_in])

        return demo


if __name__ == "__main__":
    demo = build_interface()
    logger.info("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


