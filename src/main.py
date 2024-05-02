import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from processing.CustomDataset import CustomDataset
from processing.heatmapGenerator import HeatmapGenerator
from Attribution_methods.gradcam import GradCAM
from train import Trainer  # Assurez-vous que Trainer est bien importé depuis train.py


def main():

    # Pre process
    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(project_root, 'misc', 'DATASET', 'train')
    heatmaps_path = os.path.join(project_root, 'misc', 'heatmaps')

    # Check if the heatmaps directory exists and is not empty
    if not os.path.exists(heatmaps_path) or not os.listdir(heatmaps_path):
        os.makedirs(heatmaps_path, exist_ok=True)  # Create directory if it doesn't exist
        # Initialize the HeatmapGenerator
        heatmap_generator = HeatmapGenerator(
            heatmap_dir=heatmaps_path,
            model_path=os.path.join(project_root, 'src', 'processing', 'face_landmarker.task')
        )
        heatmap_generator.generate_heatmaps_for_directory(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 75
    learning_rate = 1e-4

    # Charger le modèle ResNet50
    model = models.resnet50(pretrained=True).to(device)
    model.train()

    # Optimiseur
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Fonction de perte
    criterion = torch.nn.CrossEntropyLoss()

    # Définir les transformations pour le CustomDataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    image_dir = os.path.join(project_root, 'misc', 'DATASET', 'train')
    train_dataset = CustomDataset(image_dir=image_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # carte d'attribution utliser dans la custom loss function PAL

    trainer = Trainer(model, train_loader, criterion, optimizer, device, num_epochs, 'grad_paper')
    trainer.train()

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "resnet50_pal.pth")


if __name__ == "__main__":
    main()
