# clip_classifier.py

import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data, ITALIAN_TO_ENGLISH, CLASS_ITALIAN # Import necessary components

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# English prompts for CLIP
CLIP_PROMPTS = ["a photo of an elephant", "a photo of a dog", "a photo of a cow", "a photo of a horse"]
EVAL_CLASSES = list(ITALIAN_TO_ENGLISH.values())

def run_clip_and_evaluate(image_file, image_labels_italian):
    """Runs the CLIP zero-shot classification and evaluates results."""
    
    if not image_file:
        print("No images loaded. Exiting evaluation.")
        return

    print(f"Loading CLIP ViT-B/32 on {DEVICE}...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

    predict_raw = []
    text_tokens = clip.tokenize(CLIP_PROMPTS).to(DEVICE)
    
    print("Starting zero-shot inference...")
    for file_path in image_file:
        try:
            image = preprocess(Image.open(file_path)).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits_per_image, _ = model(image, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
          
            pos = np.argmax(probs[0])
            predicted_prompt = CLIP_PROMPTS[pos]
            predict_raw.append(predicted_prompt)
        except Exception as e:
            # print(f"Error processing {file_path}: {e}")
            predict_raw.append("Error")

    # 1. Convert Italian labels to English simple class names
    true_labels_english = [ITALIAN_TO_ENGLISH.get(label, 'Unknown') for label in image_labels_italian]

    # 2. Convert predicted prompts back to simple English class names
    predicted_classes_simple = []
    for prompt in predict_raw:
        matched = 'Unknown'
        for i, p in enumerate(CLIP_PROMPTS):
            if prompt == p:
                matched = EVAL_CLASSES[i] 
                break
        predicted_classes_simple.append(matched)

    # 3. Final Evaluation
    valid_indices = [i for i, p in enumerate(predicted_classes_simple) if p != 'Unknown' and true_labels_english[i] != 'Unknown']
    true_valid = [true_labels_english[i] for i in valid_indices]
    pred_valid = [predicted_classes_simple[i] for i in valid_indices]

    print("\n--- Zero-Shot Classification Results ---")
    print(f"Total valid samples for evaluation: {len(true_valid)}")
    
    if len(true_valid) > 0:
        accuracy = accuracy_score(true_valid, pred_valid)
        print(f"Accuracy is {accuracy:.4f}")
        
        confusion_mat = confusion_matrix(y_true=true_valid, y_pred=pred_valid, labels=EVAL_CLASSES)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=EVAL_CLASSES, yticklabels=EVAL_CLASSES)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('CLIP Zero-Shot Confusion Matrix')
        plt.show()
    else:
        print("No valid data to calculate metrics.")


if __name__ == "__main__":
    # 1. Load data from the dedicated loader file
    image_files, image_labels_italian = load_data()
    
    # 2. Run classification and evaluation
    run_clip_and_evaluate(image_files, image_labels_italian)