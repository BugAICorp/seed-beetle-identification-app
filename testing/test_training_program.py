""" test_training_program.py """
import sys
import os
import unittest
import json
from unittest.mock import MagicMock, patch, mock_open
from io import BytesIO
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from training_program import TrainingProgram

class DummyModel(nn.Module):
    """
    DummyModel class used for replacing the model in testing.
    """
    def forward(self, x):
        """ Simulate a model’s forward pass by returning random logits. """
        # Return logits for 2 classes, batch size matches input
        return torch.randn(x.size(0), 2)

class TestTrainingProgram(unittest.TestCase):
    """
    Unit testing for training program
    """
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
        self.training_program = TrainingProgram(self.mock_dataframe, "Species", 15)

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

        result = self.training_program.get_train_test_split(df)
        train_x, test_x, train_y, test_y = result

        # Check data types
        self.assertIsInstance(train_x, object)
        self.assertIsInstance(test_x, object)
        self.assertIsInstance(train_y, object)
        self.assertIsInstance(test_y, object)

    @patch('training_program.DataLoader')
    def test_training_evaluation_resnet(self, mock_loader):
        """ Test the caudal training and evaluation function """
        # Mock DataLoader
        mock_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1])),  # 4 samples
        (torch.randn(4, 3, 224, 224), torch.tensor([1, 0, 1, 0]))   # Another 4 samples
        ])
        mock_loader.__len__.return_value = 2  # Two batches

        self.training_program.training_evaluation_resnet(1, mock_loader, mock_loader, "caud")

    def test_train_resnet_model(self):
        """ Test train_resnet_model method """
       # Mock dataset with multiple samples
        mock_train_x = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
        mock_test_x = ["img5.jpg", "img6.jpg"]
        mock_train_y = [0, 1, 0, 1]
        mock_test_y = [1, 0]

        # Mock DataLoader
        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.__iter__.return_value = iter([(torch.randn(2, 3, 224, 224),
                                                torch.tensor([0, 1]))])
        mock_loader.__len__.return_value = 2  # Mocked DataLoader length
        # Mock train-test split
        self.training_program.get_train_test_split = MagicMock(
            return_value=[mock_train_x, mock_test_x, mock_train_y, mock_test_y])
        # Mock evaluation function
        self.training_program.training_evaluation_resnet = MagicMock()
        # Run train_caudal
        self.training_program.train_resnet_model(1, "caud")
        # Ensure training_evaluation_caudal was called once
        self.training_program.training_evaluation_resnet.assert_called_once()

    def test_k_fold_resnet(self):
        """ Test the k_fold_caudal method with mocked components. """
        # Repeat labels so that each class has at least k=2 samples
        mock_labels = ["GenusA", "GenusA", "GenusB", "GenusB", "GenusA", "GenusA", "GenusB", "GenusB"]
        mock_images = [f"img{i}.jpg" for i in range(len(mock_labels))]

        self.training_program.class_column = "Genus"
        self.training_program.class_string_dictionary = {"GenusA": 0, "GenusB": 1}

        self.training_program.subsets["caud"] = pd.DataFrame({
            "Genus": mock_labels,
            "Image": mock_images
        })

        # Mocks for transformations and model methods
        self.training_program.transformations = {"caud": MagicMock()}
        self.training_program.load_model = MagicMock(return_value=MagicMock())
        self.training_program.training_evaluation_resnet = MagicMock()
        self.training_program.model_accuracies = {}

        self.training_program.k_fold_resnet(num_epochs=1, view="caud", k_folds=2)

        assert self.training_program.training_evaluation_resnet.call_count == 2
        assert self.training_program.load_model.call_count == 2

    @patch("torch.nn.CrossEntropyLoss", return_value=MagicMock())
    def test_hyperparameter_training_evaluation(self, mock_loss_fn):
        """ Test the hyperparameter_training_evaluation method with mocked components. """
        # Use the DummyModel instead of MagicMock for model
        dummy_model = DummyModel()
        dummy_model.eval = MagicMock()
        dummy_model.train = MagicMock()
        dummy_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
        dummy_model.to = MagicMock(return_value=dummy_model)

        self.training_program.models["caud"] = dummy_model
        self.training_program.device = torch.device("cpu")

        # Mock DataLoader with 1 batch for train and test
        train_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1]))
        ]
        test_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([1, 0, 1, 0]))
        ]

        f1 = self.training_program.hyperparameter_training_evaluation(
            num_epochs=1,
            train_loader=train_loader,
            test_loader=test_loader,
            view="caud",
            lr=0.001,
            optimizer_type="adam"
        )

        self.assertIsInstance(f1, float)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)

    @patch("torch.utils.data.DataLoader")
    @patch("torchvision.transforms.Compose")
    def test_objective(self, mock_compose, mock_dataloader):
        """ Test the objective method with mocked components. """
        # Setup mock trial
        trial = MagicMock()
        trial.suggest_float.side_effect = [0.001, 0.2, 0.1]
        trial.suggest_categorical.side_effect = [[16, 32, 64][1], ["adam", "sgd"][0]]
        trial.suggest_int.return_value = 15

        # Setup mock dataset for stratified k-fold
        self.training_program.subsets["caud"] = [(torch.rand(3, 224, 224), i % 2) for i in range(6)]

        # Patch other methods
        self.training_program.get_subset = MagicMock(
            side_effect=lambda ds, idx: [ds[i] for i in idx]
        )
        self.training_program.load_model = MagicMock(return_value=MagicMock())
        self.training_program.hyperparameter_training_evaluation = MagicMock(return_value=0.75)
        self.training_program.create_train_transformations = MagicMock(return_value=MagicMock())

        avg_f1 = self.training_program.objective(trial, view="caud", num_epochs=1, k_folds=2)
        
        self.assertAlmostEqual(avg_f1, 0.75)
        self.assertEqual(self.training_program.hyperparameter_training_evaluation.call_count, 2)

    @patch("optuna.create_study")
    def test_run_optuna_study(self, mock_create_study):
        """ Test the run_optuna_study method with mocked components. """
        mock_study = MagicMock()
        mock_study.best_value = 0.85
        mock_study.best_params = {"lr": 0.001, "batch_size": 32}
        mock_create_study.return_value = mock_study

        # Patch objective
        self.training_program.objective = MagicMock(return_value=0.85)

        # Replace optimize with a function that calls the mocked objective
        mock_study.optimize = lambda func, n_trials: func(MagicMock())  # simulate one trial

        best_params = self.training_program.run_optuna_study(view="caud", n_trials=1)

        self.assertEqual(best_params, mock_study.best_params)
        self.assertEqual(self.training_program.objective.call_count, 1)

    @patch("torch.save")
    def test_save_models(self, mock_torch_save):
        """ Test that save_models writes to proper files """

        # Mock previous model accuracies
        mock_accuracy_dict = {
            "caud": 0.6,
            "dors": 0.4,
            "fron": 0.7,
            "late": 0.8,
        }

        self.training_program.model_accuracies = {
            "caud": 0.5, # worse than previous
            "dors": 0.6, # improved
            "fron": 0.8, # improved
            "late": 0.8, # same
        }

        # Call the function with mocked json accuracy dump
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_accuracy_dict))):
            self.training_program.save_models(
                {
                    "caud": "caud.pth",
                    "dors": "dors.pth",
                    "fron": "fron.pth",
                    "late": "late.pth",
                },
                "height.txt",
                "dict.json",
                "test_accuracies.json"
            )

        # Verify torch.save is called for each model, ignoring exact state_dict() content
        expected_calls = [
            ((unittest.mock.ANY, "dors.pth"),),
            ((unittest.mock.ANY, "fron.pth"),),
        ]
        mock_torch_save.assert_has_calls(expected_calls, any_order=True)

if __name__ == "__main__":
    unittest.main()
