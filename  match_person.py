import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pymongo import MongoClient
from siamese_model import SiameseNetwork
import numpy as np

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork(embedding_dimension=256, backbone_name='resnet50').to(device)
model.load_state_dict(torch.load("siamese_model_3_improved(0.8958).pth", map_location=device))
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === MongoDB Setup ===
client = MongoClient("mongodb://localhost:27017")
db = client["dental_identity"]
collection = db["persons"]

threshold = 0.8958  # Your Siamese threshold

# === Core Matching Logic ===
def identify(uploaded_img):
    if uploaded_img is None:
        return None, "### ❌ No image provided"

    query_img = uploaded_img.convert("RGB")
    query_tensor = transform(query_img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model.forward_once(query_tensor).squeeze().cpu()

    best = None
    best_distance = float("inf")

    for person in collection.find():
        embedding = person.get("embedding")
        if embedding:
            db_embedding = torch.tensor(embedding, dtype=torch.float32)
            distance = F.pairwise_distance(query_embedding.unsqueeze(0), db_embedding.unsqueeze(0)).item()
            if distance < best_distance:
                best_distance = distance
                best = person

    if best and best_distance < threshold:
        try:
            match_img = Image.open(best["xrayImagePath"]).convert("RGB").resize((320, 320))
        except:
            match_img = Image.new("RGB", (320, 320), color="gray")

        result_text = f""" Name: {best['name']}
**Age**: {best['age']}  
**Sex**: {best['sex']}  
**Region**: {best['region']}  
**Distance**: {best_distance:.4f}  
**Prediction**: ✅ MATCH
#### 📞 Contact Info  
**Phone**: {best['contactInfo']['phone']}  
**Email**: {best['contactInfo']['email']}  
**Address**: {best['contactInfo']['address']}  
"""
        return match_img, result_text
    else:
        return None, "### ❌ No Match Found"

# === Gradio Interface ===
demo = gr.Interface(
    fn=identify,
    inputs=gr.Image(type="pil", label="Upload Dental X-ray", width=320, height=320),
    outputs=[
        gr.Image(label="Matched X-ray", type="pil", width=320, height=320),
        gr.Markdown()
    ],
    title="🦷 Gum Biometric Identity System",
    description="Upload a dental X-ray. The system compares it with the database and finds the best match using AI.",
    allow_flagging="never",
    theme="dark"
)

if __name__ == "__main__":
    demo.launch()
