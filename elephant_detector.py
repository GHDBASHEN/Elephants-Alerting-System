# reusable_detector.py
# 
# Purpose: Loads the exported weights and performs inference for elephant detection.

import torch
import clip
from PIL import Image
import os
import sys

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
MODEL_NAME = "ViT-B/32"
# Must match the file name used in exporter.py
INPUT_WEIGHTS_FILE = "clip_detector_weights.pt" 
BINARY_PROMPTS = ["An elephant", "Another animal"]
ALERT_THRESHOLD = 0.95

# --- Model Loading and Initialization ---

def initialize_model(model_name: str, weights_path: str, device: str):
    """Initializes CLIP model structure and loads weights from the local file."""
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Error: Model weights file not found at {weights_path}. "
            "Please run 'exporter.py' first to create this file."
        )

    print(f"Initializing CLIP model structure ({model_name})...")
    # 1. Initialize the empty model structure (no download needed here)
    # Using device="cpu" for initialization is safer before mapping state_dict
    model, preprocess = clip.load(model_name, device="cpu", download_root="/tmp") 

    # 2. Load the saved weights
    print(f"Loading parameters from {weights_path} onto {device}...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    # 3. Finalize and move to target device
    model.to(device).eval()
    return model, preprocess


# Initialize the model once when the script starts
try:
    model, preprocess = initialize_model(MODEL_NAME, INPUT_WEIGHTS_FILE, DEVICE)
    print(f"✅ Model ready for inference on {DEVICE}.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize model. Details: {e}")
    sys.exit(1)


def get_elephant_alert(image_path: str) -> tuple:
    """Analyzes an image using the locally loaded CLIP model."""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False, 0.0

    try:
        # Preprocess and prepare image and text inputs
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        text_tokens = clip.tokenize(BINARY_PROMPTS).to(DEVICE)

        with torch.no_grad():
            logits_per_image, _ = model(image, text_tokens)
            probabilities = torch.softmax(logits_per_image, dim=-1).cpu().numpy()[0]

        # "An elephant" is the first prompt (index 0)
        elephant_prob = probabilities[0]
        alert_status = elephant_prob >= ALERT_THRESHOLD
        
        return alert_status, elephant_prob

    except Exception as e:
        print(f"An error occurred during detection for {image_path}: {e}")
        return False, 0.0

# --- Example Usage (Main execution block) ---
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        # Fallback to test image from the dataset if available
        try:
            default_path = './data/raw-img/elefante/'
            test_image_file = os.listdir(default_path)[0]
            test_image_path = os.path.join(default_path, test_image_file)
        except:
            test_image_path = "path/to/your/sample_test_image.jpg" 
            print("Warning: Please provide a valid path or ensure data is unzipped.")
            
    print(f"\n--- Analyzing Image: {test_image_path} ---")
    alert, probability = get_elephant_alert(test_image_path)

    if alert:
        print(f"🚨 **ALERT! Elephant Detected!**")
        print(f"Confidence: {probability:.4f} (Threshold: {ALERT_THRESHOLD})")
    else:
        print(f"✅ All Clear.")
        print(f"Elephant Confidence: {probability:.4f} (Below threshold: {ALERT_THRESHOLD})")