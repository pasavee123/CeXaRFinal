## Production Deployment Checklist

### Prerequisites
- [ ] runtime.txt is set to python-3.10
- [ ] requirements installed without errors (including timm==0.9.16)
- [ ] OPENAI_API_KEY configured as environment secret (not committed)
- [ ] **Valid EVA-X model weights available** (HuggingFace Hub ID or local path)
- [ ] EVA-X[repo] directory structure intact for model architecture

### Testing & Validation
- [ ] App runs locally with `python app.py` (will error without valid weights - expected)
- [ ] Valid model checkpoint specified in CEXAR_MODEL environment variable
- [ ] Uploading a sample X-ray returns authentic EVA-X predictions (not fallback)
- [ ] Grad-CAM overlay works correctly with Vision Transformer architecture
- [ ] Thai explanation generated successfully
- [ ] No API keys printed in logs/UI
- [ ] Tests acknowledge no-fallback behavior: `pytest -q`

### Deployment
- [ ] Space created on Hugging Face with hardware set (CPU/GPU recommended)
- [ ] Secrets configured in Space: OPENAI_API_KEY, CEXAR_MODEL
- [ ] Model weights accessible via HuggingFace Hub ID or uploaded to Space
- [ ] README updated with required model weights and deployment notes
- [ ] Error handling tested (app fails gracefully without valid weights)

### ⚠️ CRITICAL REQUIREMENTS
- **AUTHENTIC EVA-X ONLY**: App will FAIL without genuine EVA-X models - NO EXCEPTIONS
- **Repository Structure Required**: EVA-X[repo]/classification/models/ must be complete
- **Medical Safety**: Zero tolerance for fallback models in medical diagnosis
- **Performance**: Model caching reduces inference time significantly

### 🚨 FAILURE CONDITIONS (By Design)
- Missing EVA-X repository files → IMMEDIATE FAILURE
- Invalid/missing model weights → IMMEDIATE FAILURE  
- EVA-X import errors → IMMEDIATE FAILURE
- This ensures medical-grade reliability

