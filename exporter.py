# exporter.py
# 
# Purpose: Downloads the CLIP ViT-B/32 model and saves its weights
# to a local file for future offline use.

import torch
import clip
import os
import sys

# --- Configuration ---
MODEL_NAME = "ViT-B/32"
# The name of the file that will contain your reusable model weights.
OUTPUT_WEIGHTS_FILE = "clip_detector_weights.pt" 
DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"

def export_clip_weights(model_name: str, output_path: str, device: str):
    """Downloads the model and saves its state_dict locally."""
    
    if os.path.exists(output_path):
        print(f"✅ Weights file already exists at {output_path}. Skipping download.")
        print("You can proceed directly to using 'reusable_detector.py'.")
        return

    print(f"--- Exporting CLIP Model ({model_name}) ---")
    print(f"Loading and downloading model on {device}...")
    
    try:
        # Load the model. This is where the weights are downloaded to your cache.
        model, _ = clip.load(model_name, device=device)
        model.eval()

        # Save the model's parameters (state dictionary)
        torch.save(model.state_dict(), output_path)
        
        print(f"\n🎉 Success! Model weights exported to: {output_path}")
        print("This file contains your reusable model parameters.")

    except Exception as e:
        print(f"\nCRITICAL ERROR during export: {e}")
        sys.exit(1)

if __name__ == "__main__":
    export_clip_weights(MODEL_NAME, OUTPUT_WEIGHTS_FILE, DEVICE)