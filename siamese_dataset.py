# siamese_dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SiameseDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # We need to map person_ids to a continuous range for embedding grouping
        self.person_id_to_label = {pid: i for i, pid in enumerate(self.data_frame['person_id'].unique())}
        self.labels = [self.person_id_to_label[pid] for pid in self.data_frame['person_id']]

    def __len__(self):
        return len(self.data_frame)

    def _get_image(self, img_path_relative):
        full_path = os.path.join(self.image_dir, img_path_relative)
        img = Image.open(full_path).convert("RGB") # Ensure RGB for ResNet
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        # The training CSV has 'path' and 'person_id'
        img_path_relative = self.data_frame.iloc[idx]['path']
        person_id_original = self.data_frame.iloc[idx]['person_id']
        
        # Convert original person_id to our internal continuous label
        label = self.labels[idx] 
        
        img = self._get_image(img_path_relative)
        
        # For SiameseDataset, we return the image and its person_id (as a numerical label)
        return img, label