import torch
import clip
from PIL import Image
import os
import sys
import cv2
import numpy as np

# --- 1. Configuration (MUST MATCH exporter.py) ---
# NOTE: Using 'cuda' is essential for good real-time performance.
DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
MODEL_NAME = "ViT-B/32"
INPUT_WEIGHTS_FILE = "clip_detector_weights.pt" 
BINARY_PROMPTS = ["An elephant", "Another animal"]
ALERT_THRESHOLD = 0.95

# --- Motion Detection Configuration ---
MOTION_THRESHOLD_AREA = 1000 # Minimum area (in pixels) for a contour to be considered motion
BACKGROUND_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
# The MOG2 subtractor handles adapting to slow changes in lighting/background

# --- 2. Model Initialization ---

def initialize_model(model_name: str, weights_path: str, device: str):
    """Initializes CLIP model structure and loads weights from the local file."""
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Error: Model weights file not found at {weights_path}. "
            "Please run 'exporter.py' first to create this file."
        )

    # 1. Initialize the empty model structure
    # Use 'cpu' for loading to avoid immediate memory issues if the GPU is tight
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
    print(f"✅ CLIP Model ready on {DEVICE}. Alert Threshold: {ALERT_THRESHOLD}")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize model. Details: {e}")
    sys.exit(1)


# --- 3. CLIP Inference Function ---

def check_for_elephant(cv2_frame: np.ndarray) -> tuple:
    """Analyzes a single OpenCV frame for an elephant using the CLIP model."""
    
    # Convert OpenCV BGR image (numpy array) to PIL RGB image
    rgb_image = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    try:
        image = CLIP_PREPROCESS(pil_image).unsqueeze(0).to(DEVICE)
        text_tokens = clip.tokenize(BINARY_PROMPTS).to(DEVICE)

        with torch.no_grad():
            # Get logits (raw scores)
            logits_per_image, _ = CLIP_MODEL(image, text_tokens)
            # Convert logits to probabilities using softmax
            probabilities = torch.softmax(logits_per_image, dim=-1).cpu().numpy()[0]

        elephant_prob = probabilities[0] # Probability for the first prompt ("An elephant")
        alert_status = elephant_prob >= ALERT_THRESHOLD
        
        return alert_status, elephant_prob

    except Exception as e:
        # print(f"Error during CLIP inference: {e}")
        return False, 0.0


# --- 4. Main CCTV Processing Loop ---

def analyze_cctv_stream(video_source: str or int):
    """Detects motion in a live stream and checks moving frames for elephants."""

    # 1. Open the video source (camera index or stream URL)
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"CRITICAL ERROR: Could not open video source '{video_source}'.")
        print("Check if the camera is connected, the index is correct, or the RTSP/URL is valid.")
        return

    frame_count = 0
    elephant_detected_in_session = False
    
    print(f"\n--- Starting Real-Time Analysis on Source: {video_source} ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nStream lost or failed to read frame. Attempting to restart...")
            # Simple restart attempt for live streams (may need more robust logic for production)
            cap.release()
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print("Failed to re-open stream. Exiting.")
                break
            continue
            
        frame_count += 1
        original_frame = frame.copy() # Keep a copy for drawing output

        # 1. Background Subtraction to get the foreground mask
        fgmask = BACKGROUND_SUBTRACTOR.apply(frame)
        
        # 2. Clean up the mask (remove noise, fill holes)
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.dilate(fgmask, None, iterations=2)
        
        # 3. Find contours (shapes) of moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_found = False
        current_frame_alert = False
        
        for contour in contours:
            # Filter out small contours that are likely noise
            if cv2.contourArea(contour) > MOTION_THRESHOLD_AREA:
                motion_found = True
                (x, y, w, h) = cv2.boundingRect(contour)
                
                # Draw green box for general motion
                cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Check the frame with the CLIP model
                alert, prob = check_for_elephant(frame)
                
                if alert:
                    # Draw a red box for alert
                    cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                    alert_text = f"ELEPHANT! ({prob:.2f})"
                    cv2.putText(original_frame, alert_text, (x, y - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    print(f"🚨 ALERT! Frame {frame_count}: Elephant detected (Confidence: {prob:.4f})")
                    current_frame_alert = True
                    elephant_detected_in_session = True
        
        # Display session status
        status_color = (0, 0, 255) if elephant_detected_in_session else (255, 255, 0)
        status_text = f"F: {frame_count} | Motion: {motion_found} | Elephant ALERT: {elephant_detected_in_session}"
        cv2.putText(original_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Display the output (Requires a GUI environment)
        cv2.imshow("CCTV Elephant Detection", original_frame)
        # cv2.imshow("Foreground Mask (Motion)", fgmask) # Optional: uncomment for debugging
        
        # Exit condition: Press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
             print("\nExiting stream analysis upon user request.")
             break

    # --- Cleanup and Summary ---
    cap.release()
    cv2.destroyAllWindows()
    
    if elephant_detected_in_session:
        print("\n\n*** FINAL SESSION RESULT: ELEPHANT WAS DETECTED ***")
    else:
        print("\n\n*** FINAL SESSION RESULT: No elephant alerts recorded. ***")


# --- Main Execution ---
if __name__ == "__main__":
    
    # Default to the primary webcam if no argument is provided
    video_source = 0 
    
    if len(sys.argv) > 1:
        source_input = sys.argv[1]
        
        # 1. Check if the input is a file path
        if os.path.exists(source_input):
            video_source = source_input
            print(f"Using video file: {video_source}")
        else:
            # 2. Try to convert to integer (Camera Index)
            try:
                video_source = int(source_input)
                print(f"Using local camera index: {video_source}")
            # 3. Assume it's a URL (RTSP/HTTP stream)
            except ValueError:
                video_source = source_input
                print(f"Using stream URL: {video_source}")
                
    else:
        print("No source provided. Defaulting to local camera index 0 (Primary Webcam).")
        print("\nUsage examples:")
        print("  Local Webcam: python cctv_elephant_alert.py 0")
        print("  RTSP Stream:  python cctv_elephant_alert.py 'rtsp://user:pass@ip:port/stream'")
        print("  Video File:   python cctv_elephant_alert.py 'path/to/test.mp4'")

    analyze_cctv_stream(video_source)