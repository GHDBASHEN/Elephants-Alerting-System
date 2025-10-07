# 🐘 Elephants Alerting System

An intelligent motion detection and classification system that identifies elephants in videos using background subtraction and CLIP-based visual recognition.

The Elephant Alerting System relies on the capabilities of the pre-trained CLIP (Contrastive Language-Image Pre-training) model.

The original CLIP models, including the ViT-B/32 variation used in your project, were trained by OpenAI on a massive, proprietary dataset of:

400 million (Image, Text) pairs

This colossal training data, often referred to as WebImageText (WIT), is what gives the model its powerful ability to perform zero-shot classification—meaning it can classify an image of an elephant without having been explicitly trained on a labeled "elephant" dataset.

# 🐘 Detailed System Description: Elephant Alerting System
This system is an efficient, two-stage video surveillance pipeline designed for monitoring wildlife video streams (e.g., from conservation cameras) to specifically identify the presence of elephants while minimizing computational overhead.

The core innovation is combining a lightweight, classic computer vision technique with a state-of-the-art deep learning model, ensuring that the heavy computational resources are only used when truly necessary.

1. Stage One: Motion Pre-Filtering (OpenCV)
The first stage of the system is the motion detection filter, which is responsible for analyzing the video stream and deciding whether a frame requires deeper inspection. This is implemented using the OpenCV library and a technique called Background Subtraction.

Mechanism: The system initializes a model (like MOG2) that learns the static background of the scene (e.g., trees, stationary ground).

Trigger Condition: Any significant change in the frame is flagged as motion. The system isolates moving objects (contours) and only passes the frame to the next stage if the moving object's area exceeds a pre-set MOTION_THRESHOLD_AREA.

Efficiency Benefit: This step discards the vast majority of frames containing no activity, saving significant processing time and power compared to running the complex deep learning model on every single frame.

2. Stage Two: Zero-Shot Classification (CLIP)
The second stage is the intelligent classification step, which is activated only when motion is detected. This leverages the ViT-B/32 variant of the CLIP model.

Zero-Shot Principle: Instead of traditional image classification where a model is trained on specific categories, CLIP works by comparing the image's feature vector (visual representation) to the text feature vectors (language representations) of user-defined prompts.

Binary Classification: For this elephant detection project, the model compares the video frame against two opposing binary prompts:

"An elephant"

"Another animal"

Alert Generation: The system then calculates the cosine similarity between the image vector and these two text vectors. The final classification score (confidence) for "An elephant" is checked against a high-confidence threshold (e.g., 0.95). If exceeded, a definitive Elephant Alert is triggered.

# 🧠 CLIP's Foundation: Training Data Scale
The reason this system can accurately identify an elephant using the zero-shot method is directly attributed to the massive scale of the ViT-B/32 model's original pre-training.

This colossal, weakly-labeled dataset allowed CLIP to learn a broad, generalized understanding of visual concepts and associate them with descriptive natural language, which is why it can successfully identify an elephant based only on the descriptive text prompt.

---
<img width="1003" height="371" alt="Screenshot 2025-10-07 125854" src="https://github.com/user-attachments/assets/97949d7c-e563-46e9-afc6-59b7379a6bcb" />

<img width="364" height="228" alt="Screenshot 2025-10-07 125910" src="https://github.com/user-attachments/assets/c5a5fbf1-904c-4707-8c41-8b2fff7c11e4" />


### 🖥️ Example Console Output

The script displays progress logs and generates detailed alerts whenever elephants are detected in motion:

```
[INFO] Starting analysis on video_elephants.mp4
[ALERT] Elephant detected at frame 453
[ALERT] Elephant detected at frame 902
[INFO] Video processing completed.
```
<img width="800" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/45d1ce5b-8e03-435d-b7f4-f833feb86e99" />

---

## ⚙️ Technical Details and Configuration

### **Motion Detection (cv2 and Background Subtraction)**
The `video_elephant_alert.py` script utilizes:
```python
cv2.createBackgroundSubtractorMOG2()
```
This dynamically models the background and triggers detection only when a moving object's contour area exceeds the **`MOTION_THRESHOLD_AREA`** value.

---

### **CLIP Classification**

The system employs the **ViT-B/32** model from OpenAI’s CLIP for visual classification with optimized binary prompts.  
It ensures robust detection even in low-light or noisy conditions.

---

### **Model Portability**

The CLIP model architecture (Python-based) is initialized and its weights loaded via:

```python
torch.load("clip_detector_weights.pt", map_location=device)
```

This design separates the model definition from its parameters, ensuring easy transferability between different environments (CPU or GPU).

---

## 🤝 Contributing

Contributions are welcome!  

You can:
- Open an issue to report bugs or suggest new features.  
- Submit a pull request to improve the code or add enhancements such as:
  - Refining motion detection for different weather/video conditions.
  - Implementing real-time visual display using `cv2.imshow`.
  - Adding multi-video or live webcam streaming support.

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.
