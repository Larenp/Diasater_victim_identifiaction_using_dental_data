
import datetime
from pymongo import MongoClient
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
from siamese_model import SiameseNetwork

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork(embedding_dimension=256, backbone_name="resnet50").to(device)
model.load_state_dict(torch.load("siamese_model_3_improved(0.8958).pth", map_location=device))
model.eval()

# === Image Path ===
image_path = "/Users/larenpinto/Desktop/complex_fianl/cropped_parts/full/20.png"  # ✅ Absolute path

# === Preprocess Image ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 3-channel RGB
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# === Get Normalized Embedding ===
with torch.no_grad():
    embedding_tensor = model.forward_once(input_tensor)
    embedding_tensor = F.normalize(embedding_tensor, p=2, dim=1)  # ✅ L2-normalized

embedding_array = embedding_tensor.squeeze().cpu().numpy().tolist()  # 🔢 256D

# === MongoDB Insert ===
client = MongoClient("mongodb://localhost:27017")
db = client["dental_identity"]
collection = db["persons"]

record = {
    "name": "person-15",
    "age": 74,
    "sex": "Female",
    "contactInfo": {
        "phone": "4689404567890",
        "email": "person14@example.com",
        "address": "Earth"
    },
    "xrayImagePath": image_path,
    "region": "full",
    "embedding": embedding_array,
    "createdAt": datetime.datetime.utcnow()
}

collection.insert_one(record)
print("✅ Inserted with correct L2-normalized 256D embedding")
