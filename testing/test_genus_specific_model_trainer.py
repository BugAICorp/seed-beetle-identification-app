"""test_genus_specific_model_trainer.py"""
import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from genus_specific_model_trainer import GenusSpecificModelTrainer, ImageDataset

class TestGenusSpecificModelTrainer(unittest.TestCase):
    """Test the genus specific model trainer"""
    def setUp(self):
        """
        Set up test data and initialize the TrainingProgram instance.
        """
        # Create mock binary images
        def create_mock_image_blob():
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                return buffer.getvalue()
        
        # Create a mock DataFrame for testing
        self.mock_dataframe = pd.DataFrame({
            "Genus": ["GenusA", "GenusB", "GenusC", "GenusD", "GenusE",
                      "GenusF", "GenusG", "GenusH", "GenusI", "GenusJ"],
            "Species": ["SpeciesA", "SpeciesB", "SpeciesC", "SpeciesD", "SpeciesE",
                        "SpeciesF", "SpeciesG", "SpeciesH", "SpeciesI", "SpeciesJ"],
            "UniqueID": ["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9", "ID10"],
            "View": ["CAUD", "DORS", "FRON", "LATE", "CAUD", "DORS", "FRON", "LATE", "CAUD", "DORS"],
            "Image": [create_mock_image_blob() for _ in range(10)]
        })

        # Initialize the TrainingProgram instance
        self.training_program = GenusSpecificModelTrainer(self.mock_dataframe)

        # Mock the get_subset method to use the mock DataFrame
        self.training_program.get_subset = MagicMock(side_effect=self.mock_get_subset)
    
    def mock_get_subset(self, view_type, dataframe):
        """
        Mock implementation of get_subset for testing.
        Filters the mock DataFrame based on the view_type.
        Return: pd.Datafram: subset dataframe
        """
        return dataframe[dataframe["View"] == view_type] if not dataframe.empty else pd.DataFrame()

    def test_get_train_test_split(self):
        """Test get_train_test_split returns correctly split data"""
        df = self.mock_dataframe[self.mock_dataframe["View"] == "CAUD"]
        fake_dict = {"SpeciesA" : 0, "SpeciesB" : 1, "SpeciesC" : 2, "SpeciesD" : 3, "SpeciesE" : 4,
                        "SpeciesF" : 5, "SpeciesG" : 6, "SpeciesH" : 7, "SpeciesI" : 8, "SpeciesJ" : 9
                        }

        result = self.training_program.get_train_test_split(df, fake_dict)
        train_x, test_x, train_y, test_y = result

        # Check data types
        self.assertIsInstance(train_x, object)
        self.assertIsInstance(test_x, object)
        self.assertIsInstance(train_y, object)
        self.assertIsInstance(test_y, object)
    
    def test_get_subset_filters_correctly(self):
        """Test that get_subset returns rows matching the given genus"""
        genus = "GenusA"
        subset = self.training_program.get_subset(genus, self.mock_dataframe)
        self.assertTrue((subset["Genus"] == genus).all())
        self.assertEqual(len(subset), 0)

    def test_load_model_output_layer_size(self):
        """Test that load_model returns a model with the correct output layer size"""
        model = self.training_program.load_model(num_classes=5)
        self.assertEqual(model.fc.out_features, 5)

    @patch("genus_specific_model_trainer.torch.save")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_model_writes_files(self, mock_open_file, mock_torch_save):
        """Test save_model saves model and dictionary correctly"""
        dummy_model = MagicMock()
        class_dict = {"A": 0, "B": 1}
        genus = "GenusX"
        self.training_program.save_model(dummy_model, genus, class_dict)
        mock_torch_save.assert_called_once()
        mock_open_file.assert_called_with(f"src/genus_models/{genus}_dict.json", "w")

    @patch("builtins.open", new_callable=mock_open)
    def test_update_accuracies_creates_file_if_missing(self, mock_open_file):
        """Test update_accuracies when file doesn't exist"""
        genus = "GenusA"
        accuracy_path = "fake_path.json"
        self.training_program.model_accuracies = {genus: 0.9}

        with patch("json.dump") as mock_json_dump, \
             patch("json.load", side_effect=FileNotFoundError):
            should_update = self.training_program.update_accuracies(genus, accuracy_path)

        self.assertTrue(should_update)
        mock_json_dump.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    def test_update_accuracies_updates_only_if_better(self, mock_open_file):
        """Test update_accuracies updates only if the new accuracy is better"""
        genus = "GenusA"
        path = "mock_path.json"
        self.training_program.model_accuracies = {genus: 0.8}

        existing_data = {genus: 0.5}
        with patch("json.load", return_value=existing_data.copy()), \
             patch("json.dump") as mock_json_dump:
            result = self.training_program.update_accuracies(genus, path)

        self.assertTrue(result)
        mock_json_dump.assert_called_once()

    def test_image_dataset_returns_tensor_and_label(self):
        """Test ImageDataset __getitem__ returns expected outputs"""
        img_blob = self.mock_dataframe["Image"].iloc[0]
        dataset = ImageDataset([img_blob], [0], transform=self.training_program.transformation)
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(label.item(), 0)

if __name__ == "__main__":
    unittest.main()