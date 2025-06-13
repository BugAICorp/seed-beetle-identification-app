""" test_model_loader.py """

import unittest
from unittest.mock import mock_open, patch, MagicMock
import sys
import os
from io import StringIO
import pandas as pd
import torch
from torchvision import models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_loader import ModelLoader
from model_loader import load_stack_model, load_genus_specific_model

class TestModelLoader(unittest.TestCase):
    """
    Unit testing for Model Loader 
    """
    @patch("torch.load")  # Mock torch.load to prevent actual file loading
    def test_load_model_weights(self, mock_torch_load):
        """ Test that load_model_weights correctly loads weights into the model. """
        # Create a testing instance of the ModelLoader object with test mode enabled
        weights_file_paths = {"caud": "mock_weights.pth"}
        testing_instance = ModelLoader(weights_file_paths, 15, test=True)
        testing_instance.models["caud"] = MagicMock()
        testing_instance.device = torch.device("cpu")

        # Mock the return value of torch.load
        mock_torch_load.return_value = {"mock_key": torch.tensor([1.0])}

        # Call the load_model_weights method for testing
        testing_instance.load_model_weights("caud")

        # Assert torch.load was called with the correct arguments,
        # and the load_state_dict was called on the model
        mock_torch_load.assert_called_once_with(
            "mock_weights.pth", map_location=testing_instance.device, weights_only=True)
        testing_instance.models["caud"].load_state_dict.assert_called_once_with(
            mock_torch_load.return_value)

    @patch('sys.stdout', new_callable=StringIO)
    def test_load_model_weights_file_not_found(self, mock_stdout):
        """ Test that load_model_weights handles FileNotFoundError correctly. """
        # testing instance setup
        weights_file_paths = {"caud": "non_existent_weights.pth"}
        testing_instance = ModelLoader(weights_file_paths, 15, test=True)
        testing_instance.models["caud"] = MagicMock()
        testing_instance.device = torch.device("cpu")

        # Call loadModelWeights with false file paths
        testing_instance.load_model_weights("caud")

        # Assert that the correct error message was printed
        self.assertIn("Weights File for caud Model Does Not Exist.", mock_stdout.getvalue())

    @patch.object(ModelLoader, "load_model_weights")  # Mock load_model_weights
    def test_model_initializer(self, mock_load_model_weights):
        """ Test that model_initializer correctly initializes the model. """
        # Create mock weights paths to create the testing instance for ModelLoader
        weights_file_paths = {
            "caud": "mock_weights.pth",
            "dors": "mock_weights.pth",
            "fron": "mock_weights.pth",
            "late": "mock_weights.pth"
        }
        testing_instance = ModelLoader(weights_file_paths, 15, test=True)

        # Mock the models to be ResNet instances
        for key in weights_file_paths:
            testing_instance.models[key] = MagicMock(spec=models.ResNet)

        # Call the model_initializer method for testing
        testing_instance.model_initializer()

        # Assert that load_model_weights was called for each model key and nothing else
        for key in weights_file_paths:
            mock_load_model_weights.assert_any_call(key)
        self.assertEqual(mock_load_model_weights.call_count, len(weights_file_paths))

    @patch("torch.load")
    @patch("builtins.open", new_callable=unittest.mock.mock_open,
           read_data='{"0": "A", "1": "B", "2": "C"}')
    def test_load_stack_model(self, mock_file, mock_torch_load):
        """Test that load_stack_model loads meta model properly"""
        # Create dummy input DataFrame
        df = pd.DataFrame({
            "f1": [0.1, 0.2],
            "f2": [0.2, 0.3],
            "f3": [0.3, 0.4],
            "f4": [0.4, 0.5],
            "f5": [0.5, 0.6],
            "Genus": [0, 1]
        })

        # Remove label column
        df_no_label = df.drop(columns=["Genus"])

        # Mock loading the state dictionary
        mock_state_dict = MagicMock()
        mock_torch_load.return_value = mock_state_dict

        loader = ModelLoader({}, 3, test=True)

        with patch("torch.nn.Module.load_state_dict", return_value=None):
            model = load_stack_model("Genus", df, "genus_dict.json")

        mock_file.assert_called_once_with("src/models/genus_dict.json", 'r', encoding='utf-8')
        mock_torch_load.assert_called_once_with("src/models/Genus_meta.pth")
        self.assertIsInstance(model, torch.nn.Linear)
        self.assertEqual(model.in_features, df_no_label.shape[1])
        self.assertEqual(model.out_features, 3)

    @patch("model_loader.open", new_callable=mock_open, read_data='{"0": "species_a", "1": "species_b"}')
    @patch("model_loader.torch.load")
    @patch("model_loader.models.resnet50")
    def test_load_genus_specific_model_success(self, mock_resnet50, mock_torch_load, mock_file):
        # Mock model and its fc layer
        mock_model = MagicMock()
        mock_model.fc = torch.nn.Linear(2048, 100)  # dummy layer
        mock_model.fc.in_features = 2048
        mock_resnet50.return_value = mock_model
        
        device = torch.device("cpu")
        genus = "TestGenus"

        model, class_dict = load_genus_specific_model(genus, device)

        # Assert model and dict are returned
        self.assertIsNotNone(model)
        self.assertIsInstance(class_dict, dict)
        self.assertEqual(class_dict[0], "species_a")
        self.assertEqual(class_dict[1], "species_b")

        # Assert model was put into eval mode
        model.eval.assert_called_once()

        # Assert torch.load was called correctly
        mock_torch_load.assert_called_once_with(
            f"src/genus_models/{genus}_species.pth", map_location=device, weights_only=True
        )

        # Check open was called for the correct JSON file
        mock_file.assert_called_once_with(f"src/genus_models/{genus}_dict.json", 'r', encoding='utf-8')

if __name__ == "__main__":
    unittest.main()
