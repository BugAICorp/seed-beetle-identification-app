""" test_ood_tester.py """

import os
import sys
import unittest
import torch
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ood_tester import OODTester

class DummyModel(torch.nn.Module):
    """ A dummy model class that outputs random logits for testing. """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.num_classes)

class TestOODTester(unittest.TestCase):
    """ Unit tests for the OODTester class. """

    def setUp(self):
        """ Set up a dummy model and DataFrames for testing. """
        self.model = DummyModel(num_classes=5)

        def create_dummy_image_blob():
            img = Image.new('RGB', (224, 224), color=(255, 0, 0))
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                return buffer.getvalue()

        # Create dummy ID and OOD DataFrames
        id_images = [create_dummy_image_blob() for _ in range(10)]
        ood_images = [create_dummy_image_blob() for _ in range(10)]

        self.id_df = pd.DataFrame({'Image': id_images})
        self.ood_df = pd.DataFrame({'Image': ood_images})

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.tester = OODTester(self.model, self.id_df, self.ood_df, transform=transform)

    def test_compute_energy_shape(self):
        """ Test that compute_energy returns the correct shape. """
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]])
        temperature = 1.0
        energy = self.tester.compute_energy(logits, temperature)
        # Energy output should have shape (batch_size,)
        self.assertEqual(energy.shape, (2,))

    def test_get_energy_scores_type_and_length(self):
        """ Test get_energy_scores returns a numpy array of correct length. """
        energies = self.tester.get_energy_scores(self.tester.id_loader, temperature=1.0)
        self.assertIsInstance(energies, np.ndarray)
        self.assertEqual(len(energies), 10)

    def test_test_ood_returns_structure(self):
        """ Test that test_ood returns best_temp as float and correct result structure. """
        best_temp, results = self.tester.test_ood(temperatures=[1.0])
        self.assertIsInstance(best_temp, float)
        # Make sure results dictionary is correct
        self.assertIn(1.0, results)
        self.assertIn('id_energies', results[1.0])
        self.assertIn('ood_energies', results[1.0])
        self.assertIn('auroc', results[1.0])
        self.assertIn('aupr', results[1.0])

    def test_plot_distributions_runs(self):
        """ Test that plot_distributions runs without raising an exception. """
        id_energies = np.random.normal(-10, 1, size=10)
        ood_energies = np.random.normal(-5, 1, size=10)
        try:
            self.tester.plot_distributions(id_energies, ood_energies, temperature=1.0)
        except Exception as e:
            self.fail(f"plot_distributions raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
