import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import torch


def apply_transform(image, heatmap, transform):
    seed = np.random.randint(2147483647)
    random.seed(seed)
    torch.manual_seed(seed)
    image = transform(image)

    random.seed(seed)
    torch.manual_seed(seed)
    heatmap = transform(heatmap)

    return image, heatmap


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.image_names = []
        for emotion_dir in os.listdir(image_dir):
            for image in os.listdir(os.path.join(image_dir, emotion_dir)):
                self.image_names.append((emotion_dir, image))


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        emotion_dir, img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, emotion_dir, img_name)
        heatmap_name = os.path.splitext(img_name)[0] + '_h.png'

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        heatmap_path = os.path.join(project_root, "misc","heatmaps", heatmap_name)

        if not os.path.exists(heatmap_path):
            raise FileNotFoundError(f"Heatmap not found for {img_name}")

        image = Image.open(img_path).convert('RGB')
        heatmap = Image.open(heatmap_path).convert('RGB')
        label = emotion_dir

        if self.transform:
            image, heatmap = apply_transform(image, heatmap, self.transform)

        return image, heatmap, label

