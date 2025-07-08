""" test_model_visulaizer.py """

import sys
import os
import unittest
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_visualizer import GradCAMVisualizer

class DummyModel(nn.Module):
    """
    DummyModel class used for replacing the model in testing.
    """
    def __init__(self):
        """ Initializes a simple dummy CNN model for testing purposes. """
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        """ Defines the forward pass of the dummy model. """
        x = self.conv(x)
        self.activations = x
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


class TestGradCAMVisualizer(unittest.TestCase):
    """
    Unit testing for GradCAMVisualizer
    """
    def setUp(self):
        """
        Sets up the test with a dummy model, visualizer, and test image.
        """
        self.model = DummyModel()
        self.target_layer = self.model.conv
        self.visualizer = GradCAMVisualizer(self.model, self.target_layer)

        self.image_tensor = torch.randn(1, 3, 300, 300)
        self.original_image = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype(np.uint8))

    def test_generate_heatmap(self):
        """
        Test that generate_heatmap returns a 2D NumPy array.
        """
        heatmap = self.visualizer.generate_heatmap(self.image_tensor)
        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(len(heatmap.shape), 2)

    def test_overlay_heatmap(self):
        """
        Test that overlay_heatmap runs without error.
        """
        heatmap = self.visualizer.generate_heatmap(self.image_tensor)
        # Should not raise an exception
        self.visualizer.overlay_heatmap(heatmap, self.original_image)

    def test_save_visualization(self):
        """
        Test that save_visualization successfully saves the heatmap image.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_heatmap.png")
            self.visualizer.save_visualization(self.image_tensor[0], self.original_image, output_path)
            self.assertTrue(os.path.exists(output_path))


if __name__ == '__main__':
    unittest.main()
