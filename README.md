---
title: CeXaR
sdk: gradio
app_file: app.py
license: mit
---

CeXaR — Chest X-ray Analyzer + LLM Explanation

Overview
CeXaR is a medical imaging application that analyzes chest X-ray images using authentic EVA-X Vision Transformer models. It produces accurate predictions, Grad-CAM visualizations, and Thai explanations via an LLM. This application requires trained EVA-X model weights for reliable medical diagnosis.

Architecture
```
[ Gradio UI ]
   ├─ Upload X-ray -> preprocess -> [ Model ] -> predictions
   │                                 |            \
   │                                 +-- Grad-CAM  -> heatmap & overlay
   │
   ├─ Auto-initial chat message (assistant): Thai explanation from LLM
   └─ Chat (user ↔ assistant): uses same context-aware LLM, with streaming updates

predictions + image meta -> [ OpenAI LLM ] -> Thai explanation and chat replies (fallback if API fails)
```

Project Structure
- `app.py`: Gradio app entrypoint with model caching and robust error handling.
- `models/eva_x.py`: Authentic EVA-X Vision Transformer integration with no fallback models.
- `llm/openai_client.py`: OpenAI client with retries and deterministic fallback.
- `utils/image.py`: Image helpers and overlays.
- `utils/gradcam.py`: Optional grad-cam runner (not mandatory in flow).
- `utils/security.py`: Rate limiter and image validation.
- `test/`: Pytest unit tests and tiny fixtures.
- `EVA-X[repo]/`: Complete EVA-X reference repository for model architecture.

## ⚠️ Important Changes
**No Fallback Models**: This application now requires authentic trained EVA-X weights. It will fail gracefully with clear error messages if:
- Model weights are not found
- EVA-X architecture cannot be imported
- Invalid checkpoint files are provided

This ensures medical diagnosis reliability by preventing the use of untrained or inappropriate models.

## 🚀 Performance Improvements
- **Model Caching**: Models are cached using `@lru_cache` to avoid reloading on each request
- **Optimized Preprocessing**: Removed redundant image encoding/validation steps
- **Vision Transformer GradCAM**: Improved GradCAM implementation for ViT architectures
- **Better Error Handling**: Clear error messages for debugging and user guidance
- **Memory Efficiency**: Reduced memory usage through proper tensor management

Environment
- Python runtime: see `runtime.txt` (python-3.10)
- API keys are read from environment only. `.env` is supported in `app.py` via `python-dotenv` for local dev and is gitignored.

Local Run
```
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your_key>  # Windows: setx OPENAI_API_KEY <your_key> (new shell)
python app.py
```

Usage
1. Upload a chest X-ray (PNG/JPG <= 10 MB).
2. Optionally set `Model weights path or HF Hub ID` to a relative path like `models/weights/ckpt.pth` or a Hugging Face repo ID.
3. Choose device (CPU/GPU). GPU recommended on Spaces if available.
4. Click “วิเคราะห์ภาพ”. The app will:
   - Show predictions table, Grad-CAM overlay, explanation, and metadata
   - Auto-start the chat by posting the initial assistant explanation into the chatbot
5. Ask follow-up questions in the chat. Answers are streamed progressively.

Hugging Face Spaces Deploy
1. Create a new Space (Gradio).
2. Push this repo (excluding any heavy weights). Ensure `runtime.txt` and `requirements.txt` exist.
3. In Space Settings → Hardware: pick GPU if desired, else CPU.
4. In Space Settings → Secrets: add `OPENAI_API_KEY`.
5. Start the Space. If the LLM call fails, the app still returns a templated explanation and chat fallback.

Model Weights
- Place model weights under `models/weights/` (gitignored) and reference with a relative path, e.g., `models/weights/model.pth`.
- Or provide a Hugging Face Hub ID, e.g., `your-org/your-eva-x-checkpoint`. The loader tries `pytorch_model.bin` then `model.pth`.

Tests
```
pytest -q
```

Developer Notes (Provenance from EVA-X reference)
- Extracted/replicated pieces:
  - Preprocessing statistics (mean/std) for EVA-X classification from `EVA-X[repo]/classification/utils/datasets.py` (`mean=(0.49185243,...), std=(0.28509309,...)`).
  - General idea of classification head and Grad-CAM usage. We implemented a minimal CNN (`SimpleViTClassifier`) as a small demoable stand-in that accepts external weights.
- Not copied wholesale; only minimal constants and behavior were adopted. Heavy EVA-X code was intentionally not imported.

Security
- No secrets logged or displayed.
- Simple in-memory rate limit: 10 req/min.
- Image validation: size and file type check.

Commit message suggestion
- feat: initial CeXaR implementation (model wrapper, llm client, gradio app, tests)


