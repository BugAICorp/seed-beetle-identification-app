""" test_yolo_training_program.py """

import unittest
from unittest.mock import patch, MagicMock
import torch
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from yolo_training_program import YOLOTrainer


class TestYOLOTrainer(unittest.TestCase):
    """ 
    Unit tests for the YOLOTrainer class.
    """
    def setUp(self):
        """ Setup the arguements for the YOLOTrainer """
        self.dataset_yaml = "dataset.yaml"
        self.epochs = 3
        self.batch_size = 4
        self.img_size = 320

    @patch("yolo_training_program.YOLO")
    def test_initializer(self, mock_yolo):
        """ Test initializer works correctly. """
        # Mock YOLO instance and its .to() method
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        trainer = YOLOTrainer(
            dataset_yaml=self.dataset_yaml,
            epochs=self.epochs,
            batch_size=self.batch_size,
            img_size=self.img_size,
            device=torch.device("cpu")
        )

        # Check attributes
        self.assertEqual(trainer.dataset_yaml, self.dataset_yaml)
        self.assertEqual(trainer.epochs, self.epochs)
        self.assertEqual(trainer.batch_size, self.batch_size)
        self.assertEqual(trainer.img_size, self.img_size)
        self.assertEqual(trainer.device.type, "cpu")
        # Check that YOLO was instantiated with correct model name
        mock_yolo.assert_called_once_with("yolov8n.pt")
        # Check model.to was called with the device
        mock_model_instance.to.assert_called_once_with(trainer.device)

    @patch("yolo_training_program.YOLO")
    def test_train(self, mock_yolo):
        """ Test train calls YOLO train. """
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        trainer = YOLOTrainer(self.dataset_yaml, epochs=2, batch_size=2, img_size=256)
        trainer.train()

        # Check YOLO.train called once with correct args
        mock_model_instance.train.assert_called_once_with(
            data=self.dataset_yaml,
            epochs=2,
            batch=2,
            imgsz=256,
            device=str(trainer.device)
        )

    @patch("yolo_training_program.os.path.exists")
    @patch("yolo_training_program.os.path.getmtime")
    @patch("yolo_training_program.glob.glob")
    @patch("yolo_training_program.shutil.copy")
    @patch("yolo_training_program.YOLO")
    def test_save(self, mock_yolo, mock_copy, mock_glob, mock_getmtime, mock_exists):
        """ Test save calls shutil.copy to save model weights. """
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Simulate one training run folder
        mock_glob.return_value = ["runs/detect/train"]

        # Pretend mtime is valid
        mock_getmtime.return_value = 12345

        # Pretend best.pt exists
        mock_exists.return_value = True

        trainer = YOLOTrainer(self.dataset_yaml)
        
        trainer.save("src/models/yolov8n_whole_image.pt")

        mock_copy.assert_called_once_with(
            "runs/detect/train/weights/best.pt",
            "src/models/yolov8n_whole_image.pt"
        )

if __name__ == "__main__":
    unittest.main()
