import logging

from torch.utils.data import DataLoader

from processing.heatmapGenerator import HeatmapGenerator
from processing.preprocessing import Preprocessor
import os
import cv2
import mediapipe as mp
import numpy as np
import torch
from processing.CustomDataset import CustomDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

test_data_dir = os.path.join(project_root, 'misc', 'DATASET', 'test')
heatmap_generator = HeatmapGenerator(
    heatmap_dir=os.path.join(project_root, 'misc/heatmaps'),
    model_path=os.path.join(project_root, 'src', 'processing', 'face_landmarker.task')
)



def test_goo():
    index = 0
    for dirpath, dirnames, filenames in os.walk(test_data_dir):
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(dirpath, filename)
                heatmap_generator.process_image(image_path, filename)
                index += 1


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def test_custom_dataset(data_loader, num_samples=1):
    # Iterate through the DataLoader
    i = 0
    for images, heatmaps, labels in data_loader:
        if i >= num_samples:
            break

        # Display each image and heatmap pair
        for j in range(len(images)):
            plt.figure(figsize=(10, 5))

            # Display image
            plt.subplot(1, 2, 1)
            plt.imshow(transforms.ToPILImage()(images[j]))
            plt.title('Image')
            plt.axis('off')

            # Display heatmap
            plt.subplot(1, 2, 2)
            plt.imshow(transforms.ToPILImage()(heatmaps[j]))
            plt.title('Heatmap')
            plt.axis('off')

            plt.show()
            print(labels[j])
            i += 1


# Run the test function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
project_root = os.path.dirname(os.path.dirname(__file__))
custom_dataset = CustomDataset(image_dir=test_data_dir, transform=transform)
data_loader = DataLoader(custom_dataset, batch_size=16, shuffle=True)
test_custom_dataset(data_loader)
