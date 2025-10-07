# video_elephant_alert.py

import torch
import clip
from PIL import Image
import os
import sys
import cv2
import numpy as np

# --- 1. Configuration (MUST MATCH exporter.py) ---
DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
MODEL_NAME = "ViT-B/32"
INPUT_WEIGHTS_FILE = "clip_detector_weights.pt" 
BINARY_PROMPTS = ["An elephant", "Another animal"]
ALERT_THRESHOLD = 0.95

# --- Motion Detection Configuration ---
MOTION_THRESHOLD_AREA = 1000  # Minimum area (in pixels) for a contour to be considered motion
BACKGROUND_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
# The MOG2 subtractor handles adapting to slow changes in lighting/background

# --- 2. Model Initialization (Reuse from reusable_detector.py) ---

def initialize_model(model_name: str, weights_path: str, device: str):
    """Initializes CLIP model structure and loads weights from the local file."""
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Error: Model weights file not found at {weights_path}. "
            "Please run 'exporter.py' first to create this file."
        )

    # 1. Initialize the empty model structure
    model, preprocess = clip.load(model_name, device="cpu", download_root="/tmp") 

    # 2. Load the saved weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    # 3. Finalize and move to target device
    model.to(device).eval()
    return model, preprocess


# Initialize the model once
try:
    CLIP_MODEL, CLIP_PREPROCESS = initialize_model(MODEL_NAME, INPUT_WEIGHTS_FILE, DEVICE)
    print(f"✅ CLIP Model ready on {DEVICE}.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize model. Details: {e}")
    sys.exit(1)


# --- 3. CLIP Inference Function (Modified to accept OpenCV frame) ---

def check_for_elephant(cv2_frame: np.ndarray) -> tuple:
    """Analyzes a single OpenCV frame for an elephant using the CLIP model."""
    
    # Convert OpenCV BGR image (numpy array) to PIL RGB image
    # Note: cv2 uses BGR channel order, PIL/CLIP expects RGB
    rgb_image = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    try:
        image = CLIP_PREPROCESS(pil_image).unsqueeze(0).to(DEVICE)
        text_tokens = clip.tokenize(BINARY_PROMPTS).to(DEVICE)

        with torch.no_grad():
            logits_per_image, _ = CLIP_MODEL(image, text_tokens)
            probabilities = torch.softmax(logits_per_image, dim=-1).cpu().numpy()[0]

        elephant_prob = probabilities[0]
        alert_status = elephant_prob >= ALERT_THRESHOLD
        
        return alert_status, elephant_prob

    except Exception as e:
        # print(f"Error during CLIP inference: {e}")
        return False, 0.0


# --- 4. Main Video Processing Loop ---

def analyze_video_for_elephants(video_path: str):
    """Detects motion in a video and checks moving frames for elephants."""

    if not os.path.exists(video_path):
        print(f"CRITICAL ERROR: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"CRITICAL ERROR: Could not open video file {video_path}")
        return

    frame_count = 0
    elephant_detected = False
    
    print(f"\n--- Analyzing Video: {video_path} ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # 1. Background Subtraction to get the foreground mask
        # fgmask contains white pixels where motion is detected
        fgmask = BACKGROUND_SUBTRACTOR.apply(frame)
        
        # 2. Clean up the mask (remove noise, fill holes)
        fgmask = cv2.erode(fgmask, None, iterations=2)
        fgmask = cv2.dilate(fgmask, None, iterations=2)
        
        # 3. Find contours (shapes) of moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_found = False
        
        for contour in contours:
            # Filter out small contours that are likely noise
            if cv2.contourArea(contour) > MOTION_THRESHOLD_AREA:
                motion_found = True
                
                # OPTIONAL: Draw a box around the motion for visual confirmation
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # We found motion! Now, check the frame with the CLIP model
                alert, prob = check_for_elephant(frame)
                
                if alert:
                    print(f"\n🚨 ALERT! Elephant detected at Frame {frame_count}!")
                    print(f"Confidence: {prob:.4f} (Threshold: {ALERT_THRESHOLD})")
                    elephant_detected = True
                    # Optional: Break early if a positive detection is sufficient
                    # break 

        # Display progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}...", end='\r')
        
        # Optional: Display the frame and mask for debugging (requires GUI)
        # cv2.imshow("Detection Frame", frame)
        # cv2.imshow("Foreground Mask", fgmask)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # --- Cleanup and Summary ---
    cap.release()
    cv2.destroyAllWindows()
    
    if elephant_detected:
        print("\n\n*** FINAL RESULT: ELEPHANT FOUND IN VIDEO ***")
    else:
        print("\n\n*** FINAL RESULT: No elephant detected in moving frames. ***")


# --- Main Execution ---
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        video_file_path = sys.argv[1]
    else:
        video_file_path = "path/to/your/test_video.mp4"
        print("Please provide the path to your video file as a command-line argument.")
        print(f"Example: python video_elephant_alert.py {video_file_path}")

    analyze_video_for_elephants(video_file_path)