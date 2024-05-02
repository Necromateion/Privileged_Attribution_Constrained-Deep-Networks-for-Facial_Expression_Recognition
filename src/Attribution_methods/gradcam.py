import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def save_gradients(self, grad):
        self.gradients = grad

    def save_activations(self, act):
        self.activations = act

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.save_activations(output)
            return None

        def backward_hook(module, grad_in, grad_out):
            self.save_gradients(grad_out[0])
            return None

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        model_output = self.model(input_image)
        if target_class is None:
            target_class = model_output.argmax().item()

        self.model.zero_grad()
        class_loss = model_output[0, target_class]
        class_loss.backward()

        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]
        pooled_gradients = np.mean(gradients, axis=(1, 2))

        for i in range(gradients.shape[0]):
            activations[i] *= pooled_gradients[i]

        cam = np.mean(activations, axis=0)
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()  # Normalization
        cam = cv2.resize(cam, (224, 224))

        return cam
