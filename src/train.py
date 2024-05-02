import torch
from torch.utils.data import DataLoader
from Attribution_methods.gradcam import GradCAM
from Attribution_methods.grad_paper import Grad_Paper
import torch.nn.functional as F
from tqdm import tqdm

def normalize_map(map):
    return map / (map.max() + 1e-5)


def privileged_attribution_loss(attribution_maps, heatmap_prior):
    resized_heatmap_prior = F.interpolate(heatmap_prior, size=attribution_maps[0].shape[-2:], mode='bilinear')
    normalized_heatmap_prior = normalize_map(resized_heatmap_prior)

    # Ensure dimensions match
    assert attribution_maps[0].shape[-2:] == normalized_heatmap_prior.shape[-2:], "Mismatch in attribution map and heatmap dimensions"

    loss = sum(-torch.sum(amap * normalized_heatmap_prior) for amap in attribution_maps)
    return loss



def custom_loss_function(original_loss, attribution_maps, heatmap_prior, alpha=0.5):
    pal_loss = privileged_attribution_loss(attribution_maps, heatmap_prior)
    adjusted_loss = original_loss + alpha * pal_loss
    return adjusted_loss


class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device, num_epochs, attribution_method='gradcam'):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.attribution_method = attribution_method.lower()
        if self.attribution_method == 'gradcam':
            self.attribution = GradCAM(model, target_layer='layer4')  # example layer
        elif self.attribution_method == 'grad_paper':
            self.attribution = Grad_Paper(model, target_layers='layer4')
        else:
            raise ValueError("Invalid attribution method specified.")

    def train(self):
        for epoch in range(self.num_epochs):
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for images, heatmaps, labels in self.train_loader:
                    images, heatmaps = images.to(self.device), heatmaps.to(self.device)
                    labels = torch.tensor([int(label) for label in labels], dtype=torch.long).to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    if self.attribution_method == 'gradcam':
                        attribution_maps = [self.attribution.generate_cam(image.unsqueeze(0), target)
                                            for image, target in zip(images, outputs.argmax(1))]
                    elif self.attribution_method == 'grad_paper':
                        attribution_maps = self.attribution.generate_attribution_map(images, outputs.argmax(1))

                    if not attribution_maps:
                        print("No attribution maps generated. Check generate_attribution_map method.")
                        continue
                    #Convertir en tenseur et ajuster la perte
                    attribution_maps_tensor = torch.stack(attribution_maps).to(self.device)
                    adjusted_loss = custom_loss_function(loss, attribution_maps_tensor, heatmaps)

                    adjusted_loss.backward()
                    self.optimizer.step()
                    tepoch.set_postfix(loss=adjusted_loss.item())

                    print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {adjusted_loss.item():.4f}")