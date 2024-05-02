import torchvision.transforms as transforms
from PIL import Image
import os


class Preprocessor:
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.augmentation_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
        ])

    def process_image(self, image_path, augment=False):
        image = Image.open(image_path)
        image = self.base_transform(image)
        if augment:
            image = self.augmentation_transform(image)
        return image

    def process_directory(self, dir_path):
        processed_images = []
        for filename in os.listdir(dir_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(dir_path, filename)
                processed_images.append(self.process_image(image_path))
        return processed_images
