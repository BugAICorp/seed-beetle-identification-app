"""test_eval_species_by_genus.py"""
import unittest
import sys
import os
from unittest.mock import MagicMock, patch, mock_open
import torch
from PIL import Image
import numpy as np
import json
import dill
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from eval_species_by_genus import EvalSpeciesByGenus

class DummyModel:
    def __init__(self, output_tensor):
        self.output_tensor = output_tensor
    def to(self, device):
        return self
    def __call__(self, input_tensor):
        return self.output_tensor

class TestEvalSpeciesByGenus(unittest.TestCase):
    """
    Test the eval species by genus class
    """
    def setUp(self):
        # Dummy genus index mapping
        self.genus_idx_dict = {0: 'GenusA', 1: 'GenusB'}
        self.species_idx_dict = {0: 'SpeciesA', 1: 'SpeciesB', 2: 'SpeciesC'}

        # Dummy PIL image
        self.dummy_image = Image.new('RGB', (224, 224), color='white')

        # Dummy transformation function that returns tensor with batch dim
        self.mock_transform = lambda x: torch.rand(1, 3, 224, 224)

        self.mock_transformations = [self.mock_transform] * 4

        # Dummy output tensor for genus model (batch size 1, 2 classes)
        dummy_output = torch.tensor([[1.0, 2.0]])

        # Create dummy models with .to() and __call__
        self.mock_models = {
            'caud': DummyModel(dummy_output),
            'dors': DummyModel(dummy_output),
            'fron': DummyModel(dummy_output),
            'late': DummyModel(dummy_output),
        }

        # Patch open_class_dictionary to return genus_idx_dict
        patcher_json = patch('eval_species_by_genus.EvalSpeciesByGenus.open_class_dictionary', return_value=self.genus_idx_dict)
        patcher_transform = patch('eval_species_by_genus.EvalSpeciesByGenus.get_transformations', return_value=self.mock_transformations)

        # Dummy species model output (batch size 1, 3 classes)
        species_output = torch.tensor([[0.2, 0.5, 0.3]])

        # Patch load_genus_specific_model to return DummyModel and species_idx_dict
        def mock_load_genus_specific_model(genus, device):
            return DummyModel(species_output), self.species_idx_dict

        patcher_species = patch('eval_species_by_genus.load_genus_specific_model', side_effect=mock_load_genus_specific_model)

        self.addCleanup(patcher_json.stop)
        self.addCleanup(patcher_transform.stop)
        self.addCleanup(patcher_species.stop)

        patcher_json.start()
        patcher_transform.start()
        patcher_species.start()

        self.evaluator = EvalSpeciesByGenus(self.mock_models, 'dummy_genus.json')

    def test_transform_input_applies_transformation(self):
        result = self.evaluator.transform_input(self.dummy_image, self.mock_transform)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)  # Unsqueezed

    def test_get_genus_returns_expected_output(self):
        genus, score = self.evaluator.get_genus(caud=self.dummy_image)
        self.assertEqual(genus, 'GenusB')  # class 1 has higher logits
        self.assertIsInstance(score, float)

    def test_get_species_outputs_top_k(self):
        self.evaluator.load_species_models('GenusB')
        results = self.evaluator.get_species(caud=self.dummy_image)
        results = [(label, float(score)) for label, score in results]
        self.assertTrue(all(isinstance(label, str) and isinstance(score, float) for label, score in results))
        self.assertEqual(len(results), 5)

    def test_classify_images_combines_genus_and_species(self):
        genus_result, species_result = self.evaluator.classify_images(caud=self.dummy_image)
        self.assertIsInstance(genus_result, tuple)
        self.assertIsInstance(species_result, list)

if __name__ == '__main__':
    unittest.main()
