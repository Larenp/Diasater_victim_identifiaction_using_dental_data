# pair_evaluation_dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PairEvaluationDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.pairs_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs_frame)

    def _get_image(self, img_path_relative):
        full_path = os.path.join(self.image_dir, img_path_relative)
        img = Image.open(full_path).convert("RGB") # Ensure RGB for ResNet
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        img1_path_relative = self.pairs_frame.iloc[idx, 0]
        img2_path_relative = self.pairs_frame.iloc[idx, 1]
        label = self.pairs_frame.iloc[idx, 2] # This is the 0/1 label

        img1 = self._get_image(img1_path_relative)
        img2 = self._get_image(img2_path_relative)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)