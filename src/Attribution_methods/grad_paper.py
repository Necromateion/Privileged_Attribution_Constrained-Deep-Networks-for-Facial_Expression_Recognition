import torch


class Grad_Paper:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = [target_layers] if isinstance(target_layers, str) else target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(self.save_activation(name)))
                self.hooks.append(module.register_full_backward_hook(self.save_gradient(name)))
 




    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.clone().detach()

        return hook

    def save_gradient(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output is not None and len(grad_output) > 0:
                self.gradients[name] = grad_output[0].clone().detach()
            else:
                print(f"No gradient output for layer {name}")
        return hook


    def generate_attribution_map(self, input_tensor, target_classes):
        #print("Generating attribution maps...")

        # Ensure the model is in evaluation mode
        self.model.eval()

        self.model.zero_grad()
        outputs = self.model(input_tensor)

        #print(f"Outputs shape: {outputs.shape}")
        #print(f"Target classes: {target_classes}")

        attribution_maps = []
        for i in range(len(input_tensor)):
            #print(f"Processing input {i}")

            target = outputs[i, target_classes[i]]
            #print(f"Target (class {target_classes[i]}): {target.item()}")

            target.backward(retain_graph=True)

            for layer in self.target_layers:
                if layer not in self.gradients:
                    print(f"No gradients recorded for layer {layer}")
                    continue

                gradients = self.gradients[layer].clone()
                activations = self.activations[layer][i].clone()

                #print(f"Gradients shape for layer {layer}: {gradients.shape}")
                #print(f"Activations shape for layer {layer}: {activations.shape}")

                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
                modified_activations = torch.zeros_like(activations)
                for j in range(activations.size(0)):
                    modified_activations[j, :, :] = activations[j, :, :] * pooled_gradients[j]

                attribution_map = torch.mean(modified_activations, dim=0).detach()
                #print(f"Attribution map shape for input {i}, layer {layer}: {attribution_map.shape}")

                attribution_maps.append(attribution_map)

            self.model.zero_grad()

        #print(f"Total attribution maps generated: {len(attribution_maps)}")
        return attribution_maps



    def clean(self):
        for hook in self.hooks:
            hook.remove()
