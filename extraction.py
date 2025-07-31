import os
import cv2
from tqdm import tqdm

# === 📂 INPUT & OUTPUT ===
input_folder = "/Users/larenpinto/Desktop/all_teeth/processed_xrays"
output_root =  "/Users/larenpinto/Desktop/all_teeth/processed_xrays"

# === 🧠 Cropping Function ===
def extract_parts(img):
    h, w = img.shape
    return {
        "full": img,
           "upper": img[int(h * 0.0):int(h * 0.58), int(w * 0.05):int(w * 0.95)],
        "lower": img[int(h * 0.55):int(h * 0.95), int(w * 0.1):int(w * 0.9)],
         "gum": img[int(h * 0.25):int(h * 0.4), int(w * 0.2):int(w * 0.8)],
        
    }

# === 🔧 Make Output Folders ===
os.makedirs(output_root, exist_ok=True)
for region in ["full", "upper", "lower", "gum",]:
    os.makedirs(os.path.join(output_root, region), exist_ok=True)

# === 🔢 Natural Sort Helper ===
def extract_number(filename):
    return int(os.path.splitext(filename)[0])

# === 📦 Get Sorted Images ===
image_files = sorted(
    [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
    key=extract_number
)

# === 🚀 Begin Extraction ===
for img_name in tqdm(image_files, desc="Cropping in Order"):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Skipped unreadable: {img_name}")
        continue

    parts = extract_parts(img)
    
    for region, cropped in parts.items():
        save_path = os.path.join(output_root, region, img_name)
        cv2.imwrite(save_path, cropped)
