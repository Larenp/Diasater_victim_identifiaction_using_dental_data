import cv2
import numpy as np
import os

# === CONFIG ===
input_dir = "xrays"                  # Folder with raw X-ray images
output_dir = "processed_xrays1"       # Folder to save processed images
target_size = (224, 224)             # Model input size

os.makedirs(output_dir, exist_ok=True)

def preprocess_image(path, save_path=None, target_size=(224, 224)):
    # 🖼️ Load in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 🔳 CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 🔁 Resize using INTER_AREA (preserves details)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # ✨ Optional sharpening
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharpen_kernel)

    # 💾 Save the preprocessed image
    if save_path:
        cv2.imwrite(save_path, img)

    return img

# === Batch Process All Images ===
print("🌀 Starting preprocessing...")
count = 0
for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        preprocess_image(input_path, output_path)
        count += 1

print(f"✅ Preprocessing complete. {count} images saved to '{output_dir}'")
