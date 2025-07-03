""" test_ood_tester.py """

import os
import sys
import unittest
import torch
import numpy as np
from torch.utils.data import Dataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ood_tester import OODTester

class DummyDataset(Dataset):
    """ A dummy dataset class that generates random tensors for testing. """
    def __init__(self, length, shape):
        self.length = length
        self.shape = shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.rand(self.shape)

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
        """ Set up a dummy model and datasets for testing. """
        self.model = DummyModel(num_classes=5)
        self.id_dataset = DummyDataset(length=10, shape=(3, 224, 224))
        self.ood_dataset = DummyDataset(length=10, shape=(3, 224, 224))
        self.tester = OODTester(self.model, self.id_dataset, self.ood_dataset, batch_size=2)

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
        # Energy scores should be a numpy array
        self.assertIsInstance(energies, np.ndarray)
        # Energy scores should have one entry per sample
        self.assertEqual(len(energies), 10)

    def test_test_ood_returns_structure(self):
        """ Test that test_ood returns best_temp as float and correct result structure. """
        best_temp, results = self.tester.test_ood(temperatures=[1.0])
        self.assertIsInstance(best_temp, float)
        # assert results dictionary is correct
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
