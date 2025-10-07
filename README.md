# 🐘 Elephants Alerting System

An intelligent motion detection and classification system that identifies elephants in videos using background subtraction and CLIP-based visual recognition.

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
