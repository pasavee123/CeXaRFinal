#!/usr/bin/env python3
"""
Test strict EVA-X loading (no fallbacks)
"""
import logging
from models.eva_x import create_model

# Setup logging to see what happens
logging.basicConfig(level=logging.INFO)

def test_eva_strict():
    """Test that EVA-X loading is strict"""
    print("Testing strict EVA-X model loading...")
    
    try:
        # This should either work with EVA-X or fail completely
        model, preprocess, labels = create_model(
            model_name="test_weights", 
            device="cpu"
        )
        print("✅ SUCCESS: EVA-X model loaded successfully")
        print(f"Model type: {type(model)}")
        print(f"Number of classes: {len(labels)}")
        
    except RuntimeError as e:
        print("❌ EXPECTED FAILURE: EVA-X model loading failed")
        print(f"Error: {e}")
        print("This is expected if EVA-X repository or weights are not properly set up")
        
    except Exception as e:
        print("💥 UNEXPECTED ERROR:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_eva_strict()
