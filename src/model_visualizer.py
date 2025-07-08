""" model_visualizer.py """

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


class GradCAMVisualizer:
    """
    A class to generate Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations for
    convolutional neural networks.

    This implementation supports forward and backward hooks to capture the gradients and activations
    of a target convolutional layer, which are used to compute a class activation heatmap (CAM).
    """

    def __init__(self, model, target_layer):
        """
        Initializes the GradCAMVisualizer.

        Args:
            model (torch.nn.Module): The pretrained model to visualize.
            target_layer (torch.nn.Module): The layer in the model to visualize (usually last conv layer).
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_built()
            else 'cpu'
        )

        self.model = model.to(self.device).eval()

        # Placeholders to store gradients and activations
        self.gradients = None
        self.activations = None

        # Register hooks to the target layer for Grad-CAM
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer):
        """
        Registers forward and backward hooks on the target layer to store activations and gradients.

        Args:
            target_layer (torch.nn.Module): Layer to register hooks on.
        """
        # Save forward activations
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Save backward gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        try:
            target_layer.register_full_backward_hook(backward_hook)
        except AttributeError:
            target_layer.register_backward_hook(backward_hook)

        # Register forward hook
        target_layer.register_forward_hook(forward_hook)

    def generate_heatmap(self, input_tensor):
        """
        Generates a Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor (torch.Tensor): The input image tensor (1, C, H, W).

        Returns:
            np.ndarray: The normalized Grad-CAM heatmap.
        """
        input_tensor = input_tensor.to(self.device)

        # Zero out previous gradients and run the forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Choose the class with the highest predicted score
        class_idx = torch.argmax(output, dim=1).item()

        # Backpropagate the score of that class
        class_score = output[0, class_idx]
        class_score.backward()

        # Compute the importance weights by averaging gradients across spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Multiply weights by the activations and sum across channels to get the CAM
        cam = (weights * self.activations).sum(dim=1).squeeze()

        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)

        # Normalize heatmap to [0, 1] range
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam.cpu().numpy()

    def overlay_heatmap(self, heatmap, image, alpha=0.5):
        """
        Overlays the Grad-CAM heatmap on the original image using matplotlib.

        Args:
            heatmap (np.ndarray): Heatmap array (2D, values between 0-1).
            image (PIL.Image): Original image in PIL format.
            alpha (float): Transparency of the heatmap overlay.
        """
        # Convert heatmap to a PIL image scaled to 0–255, resize to match input image
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)

        # Convert to NumPy for matplotlib
        heatmap_np = np.array(heatmap_img)

        # Show base image and then show overlaid heatmap with color
        plt.imshow(image)
        plt.imshow(heatmap_np, cmap='jet', alpha=alpha)

        plt.axis('off')

    def save_visualization(self, image_tensor, original_image, output_path, title="Grad-CAM"):
        """
        Generates and saves the Grad-CAM overlay visualization to a file.

        Args:
            image_tensor (torch.Tensor): Preprocessed input image tensor.
            original_image (PIL.Image): The original image for overlay.
            output_path (str): Path to save the visualization.
            title (str): Title to show on the plot.
        """
        # Generate Grad-CAM heatmap
        heatmap = self.generate_heatmap(image_tensor.unsqueeze(0))

        # Create a plot and overlay the heatmap
        plt.figure()
        self.overlay_heatmap(heatmap, original_image)

        plt.title(title)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
