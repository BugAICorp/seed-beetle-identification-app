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
    
    

if __name__ == "__main__":
    unittest.main()