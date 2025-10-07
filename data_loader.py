# data_loader.py

import os

# --- Configuration for Data Loading ---
# NOTE: This path MUST match where you unzipped the dataset.
# If you downloaded to ./data and unzipped everything inside, 
# the image folders are typically in a subfolder called 'raw-img'.
UNZIPPED_DIR = './data/animals10/raw-img' 

# The folders in your dataset are Italian
CLASS_ITALIAN = ["elefante", "cane", "mucca", "cavallo"] 

# Mapping for evaluation and usage
ITALIAN_TO_ENGLISH = {
    "elefante": "Elephant",
    "cane": "Dog",
    "mucca": "Cow",
    "cavallo": "Horse"
}

def load_data():
    """
    Loads image paths and Italian labels from the dataset folders.
    
    Returns:
        tuple: (list of image file paths, list of Italian labels)
    """
    image_labels_italian = []
    image_file = []
    
    if not os.path.exists(UNZIPPED_DIR):
        print(f"Error: Data directory not found at {UNZIPPED_DIR}.")
        print("Please ensure 'animals10.zip' was successfully unzipped.")
        return [], []
        
    print(f"Loading data from: {UNZIPPED_DIR}")
    
    for i in CLASS_ITALIAN:
        paths = os.path.join(UNZIPPED_DIR, i) 
        
        if not os.path.exists(paths):
            print(f"Warning: Missing folder: {paths}. Skipping.")
            continue
            
        images = os.listdir(paths)
        
        for img in images:
            # Check for common image extensions
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_file.append(os.path.join(paths, img))
                image_labels_italian.append(i) # Use Italian label
                
    print(f"Total images loaded: {len(image_file)}")
    return image_file, image_labels_italian

if __name__ == '__main__':
    # Test the data loader
    files, labels = load_data()
    print(f"Example file: {files[0]}")
    print(f"Example label: {labels[0]}")