import torch


class GradientAttribution:
    def __init__(self, model):
        self.model = model

    def compute_gradients(self, input_image, target_class):
        input_image.requires_grad = True

        # Forward pass
        output = self.model(input_image)
        self.model.zero_grad()

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Backward pass
        output.backward(gradient=one_hot_output)

        # Grab the gradients directly from the input image
        gradients = input_image.grad.data

        return gradients
