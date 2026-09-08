# 🦷 Disaster Victim Identification Using Dental Data

A deep learning system for identifying disaster victims by matching dental X-rays using a **Siamese Neural Network** with a ResNet50 backbone. The system extracts embeddings from dental X-rays and compares them against a MongoDB database to find the closest match — enabling fast, AI-powered forensic identification.

---

## 🧠 How It Works

```
Raw X-ray → Preprocessing → Region Extraction → Embedding Generation → MongoDB Match → Identity Result
```

1. **Preprocessing** — Enhances X-ray images using CLAHE contrast normalization and sharpening.
2. **Region Extraction** — Crops the X-ray into 4 anatomical regions: `full`, `upper`, `lower`, and `gum`.
3. **Label Generation** — Produces training CSV files with online mining and balanced validation pairs.
4. **Model Training** — A Siamese Network (ResNet50 backbone, 256D embeddings) is trained on image pairs.
5. **Database Enrollment** — Victim X-rays are embedded and stored in MongoDB with personal metadata.
6. **Identification** — A query X-ray is embedded and compared against all DB records using Euclidean distance.
7. **Gradio UI** — A clean web interface for uploading X-rays and viewing identity match results.

---

## 📁 Project Structure

```
├── preprocessing.py           # X-ray image preprocessing (CLAHE, sharpening, resize)
├── extraction.py              # Crop X-rays into anatomical regions (full/upper/lower/gum)
├── labels.py                  # Generate training & balanced validation pair CSVs
├── siamese_dataset.py         # PyTorch Dataset for online mining training
├── pair_evaluation_dataset.py # PyTorch Dataset for validation pair evaluation
├── db_process.py              # Encode and insert X-ray embeddings into MongoDB
├──  match_person.py           # Gradio UI — upload X-ray and find identity match
└── README.md
```

> **Note:** `siamese_model.py` (the model architecture) is expected separately — it defines `SiameseNetwork(embedding_dimension, backbone_name)`.

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- MongoDB running locally on port `27017`
- CUDA-capable GPU (optional, but recommended)

### Install Dependencies

```bash
pip install torch torchvision opencv-python pillow pymongo gradio tqdm pandas scikit-learn numpy
```

---

## 🚀 Usage

### Step 1 — Preprocess Raw X-rays

```bash
python preprocessing.py
```
Reads images from `xrays/`, applies CLAHE + sharpening, and saves to `processed_xrays1/`.

---

### Step 2 — Extract Anatomical Regions

```bash
python extraction.py
```
Crops each X-ray into 4 regions (`full`, `upper`, `lower`, `gum`) and saves them into subfolders.

---

### Step 3 — Generate Training Labels

```bash
python labels.py
```
Produces:
- `train_online_mining.csv` — All images with person IDs (for training with online hard mining)
- `val_balanced_pairs.csv` — 1000 positive + 1000 negative pairs (for validation metrics)

---

### Step 4 — Train the Siamese Network

> Training script (`train.py` or equivalent) should use `SiameseDataset` and `PairEvaluationDataset`.
> The model file `siamese_model_3_improved(0.8958).pth` is the trained checkpoint.

---

### Step 5 — Enroll a Person into the Database

Edit `db_process.py` to set the correct `image_path` and personal details, then run:

```bash
python db_process.py
```

This generates a **256D L2-normalized embedding** and inserts the following record into MongoDB:

```json
{
  "name": "person-15",
  "age": 74,
  "sex": "Female",
  "contactInfo": { "phone": "...", "email": "...", "address": "..." },
  "xrayImagePath": "/path/to/image.png",
  "region": "full",
  "embedding": [0.012, -0.043, ...],
  "createdAt": "2024-..."
}
```

---

### Step 6 — Run the Identification UI

```bash
python " match_person.py"
```

Opens a **Gradio web interface** where you can:
- Upload a dental X-ray image
- View the matched person's X-ray and identity details
- See match distance and contact information

**Match threshold:** `0.8958` (Euclidean distance — tuned to the model's best accuracy)

---

## 🏗️ Model Architecture

| Component     | Details                                          |
|---------------|--------------------------------------------------|
| Backbone      | ResNet50 (ImageNet pretrained)                   |
| Embedding     | 256D L2-normalized                               |
| Loss          | Contrastive / Triplet (online hard mining)       |
| Input size    | 224 × 224 RGB                                    |
| Match metric  | Euclidean (pairwise) distance                    |
| Threshold     | 0.8958                                           |

---

## 🗄️ Database Schema (MongoDB)

- **Database:** `dental_identity`
- **Collection:** `persons`

Each document stores the person's identity metadata alongside their 256-dimensional dental embedding for fast nearest-neighbor lookup.

---

## 📊 Dataset Regions

Each X-ray is cropped into 4 anatomical views:

| Region  | Crop Area                          |
|---------|------------------------------------|
| `full`  | Entire X-ray                       |
| `upper` | Top 58% (upper jaw)                |
| `lower` | Bottom 40% (lower jaw)             |
| `gum`   | Central gum region (25%–40% height)|

---

## 🔍 Identification Logic

```python
# For each uploaded X-ray:
query_embedding = model.forward_once(query_tensor)

# Compare against all DB embeddings using Euclidean distance
distance = F.pairwise_distance(query_embedding, db_embedding)

# Match if below threshold
if distance < 0.8958:
    return MATCH
else:
    return NO MATCH
```

---

## 🖥️ Gradio Interface

The web UI (`match_person.py`) provides:
- **Input:** Upload a dental X-ray (PNG/JPG)
- **Output 1:** Matched X-ray image from the database
- **Output 2:** Structured markdown card with name, age, sex, region, distance score, and contact information

---

## ⚠️ Limitations

- The system currently performs a **linear scan** over all MongoDB records — consider FAISS or a vector database for large-scale deployments.
- Image quality significantly impacts match accuracy; ensure consistent X-ray acquisition protocols.
- The model checkpoint path is hardcoded — update `siamese_model_3_improved(0.8958).pth` as needed.

---

## 📄 License

This project is intended for academic and forensic research purposes.

---

## 👤 Author

Developed as part of a disaster victim identification research project using biometric dental data and deep metric learning.
