""" test_cam_training_program.py """

import os
import sys
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
from cam_training_program import CAMGuidedTrainingProgram

class DummyModel(nn.Module):
    """ Dummy model for testing CAM-guided pipeline. """
    def forward(self, x):
        return torch.randn(x.size(0), 2)

class TestCAMGuidedTrainingProgram(unittest.TestCase):
    """ Unit tests for CAMGuidedTrainingProgram. """

    def setUp(self):
        """ Set up mock dataframe and CAMGuidedTrainingProgram instance. """
        def create_mock_image_blob():
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                return buffer.getvalue()

        self.mock_dataframe = pd.DataFrame({
            "Genus": ["GenusA"] * 10,
            "Species": ["SpeciesA"] * 10,
            "UniqueID": [f"ID{i}" for i in range(10)],
            "View": ["CAUD", "DORS", "FRON", "LATE"] * 2 + ["CAUD", "DORS"],
            "Image": [create_mock_image_blob() for _ in range(10)]
        })

        self.program = CAMGuidedTrainingProgram(self.mock_dataframe, "Genus", 2, mask_dir="./masks")

    def test_get_subset(self):
        """ Test get_subset method returns filtered dataframe. """
        caud_subset = self.program.get_subset("CAUD", self.mock_dataframe)
        self.assertEqual(set(caud_subset["View"]), {"CAUD"})

    def test_get_train_test_split(self):
        """ Test get_train_test_split returns correct splits. """
        caud_df = self.program.get_subset("CAUD", self.mock_dataframe)
        split = self.program.get_train_test_split(caud_df)
        self.assertEqual(len(split), 4)

    def test_create_train_transformations(self):
        """ Test creation of training transformations with augmentation. """
        transforms = self.program.create_train_transformations()
        self.assertIn("caud", transforms)
        self.assertTrue(callable(transforms["caud"]))

    def test_load_attention_masks(self):
        """ Test loading binary masks from file paths. """
        with patch("PIL.Image.open", return_value=Image.fromarray((255 * torch.ones(300, 300).numpy()).astype("uint8"))):
            dummy_paths = ["some/path/img1.jpg", "some/path/img2.jpg"]
            masks = self.program.load_attention_masks(dummy_paths)
            self.assertEqual(masks.shape[0], len(dummy_paths))

    def test_cam_loss_shape_match(self):
        """ Test cam_loss returns scalar when masks and CAM match shape. """
        cam = torch.rand(4, 300, 300)
        mask = torch.rand(4, 300, 300)
        loss = self.program.cam_loss(cam, mask)
        self.assertTrue(torch.is_tensor(loss))

    def test_cam_loss_shape_mismatch(self):
        """ Test cam_loss when CAM and mask need interpolation. """
        cam = torch.rand(4, 300, 300)
        mask = torch.rand(4, 100, 100)
        loss = self.program.cam_loss(cam, mask)
        self.assertTrue(torch.is_tensor(loss))

    def test_load_model(self):
        """ Test load_model returns a ResNet model. """
        model = self.program.load_model()
        self.assertIsInstance(model, nn.Module)

    @patch("torch.save")
    def test_save_models_and_accuracies(self, mock_save):
        """ Test save_models saves only improved models. """
        self.program.model_accuracies = {"caud": 0.8, "dors": 0.7, "fron": 0.9, "late": 0.6}
        prev_accuracies = {"caud": 0.5, "dors": 0.7, "fron": 0.6, "late": 0.9}
        self.program.models = {k: DummyModel() for k in ["caud", "dors", "fron", "late"]}

        with patch("builtins.open", mock_open(read_data=json.dumps(prev_accuracies))):
            self.program.save_models(
                model_filenames={"caud": "caud.pth", "dors": "dors.pth", "fron": "fron.pth", "late": "late.pth"},
                height_filename="height.txt",
                class_dict_filename="class_dict.json",
                accuracy_dict_filename="accuracies.json",
                overwrite_accuracies=False
            )

        mock_save.assert_any_call(self.program.models["caud"].state_dict(), "caud.pth")
        mock_save.assert_any_call(self.program.models["fron"].state_dict(), "fron.pth")
        calls = [call.args[1] for call in mock_save.call_args_list]
        self.assertNotIn("late.pth", calls)

    @patch("cam_training_program.CAMImageDataset")
    @patch("cam_training_program.StratifiedKFold")
    @patch.object(CAMGuidedTrainingProgram, "load_model")
    @patch.object(CAMGuidedTrainingProgram, "create_train_transformations")
    @patch.object(CAMGuidedTrainingProgram, "cam_loss", return_value=torch.tensor(0.1))
    def test_cam_optuna_objective_runs(self, mock_cam_loss, mock_create_trans, mock_load_model, mock_skf, mock_dataset):
        """ Test cam_optuna_objective runs and returns a float score. """
        df = pd.DataFrame({"Genus": ["GenusA"] * 6, "View": ["CAUD"] * 6, "Image": [f"img_{i}.jpg" for i in range(6)]})
        self.program.subsets = {"caud": df}
        self.program.class_string_dictionary = {"GenusA": 0}
        self.program.image_column = "Image"
        self.program.transformations = {"caud": lambda x: x}
        mock_skf.return_value.split.return_value = [(range(3), range(3))]
        mock_dataset.side_effect = lambda *args, **kwargs: [(torch.rand(3, 3, 64, 64), torch.tensor([0, 0, 0]), ["p1", "p2", "p3"])]

        dummy_model = MagicMock()
        dummy_model.parameters.return_value = []
        dummy_model.eval = lambda: None
        dummy_model.train = lambda: None
        dummy_model.return_value = torch.randn(3, 1)
        mock_load_model.return_value = dummy_model
        self.program.models["caud"] = dummy_model

        self.program.hyperparameter_training_evaluation = lambda *args, **kwargs: 0.88

        class DummyTrial:
            """ Dummy Trial class for testing. """
            def suggest_loguniform(self, name, low, high): return 1e-3
            def suggest_categorical(self, name, choices): return choices[0]
            def suggest_int(self, name, low, high): return low
            def suggest_float(self, name, low, high): return low

        score = self.program.cam_optuna_objective(DummyTrial(), view="caud", num_epochs=1, n_splits=2)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
