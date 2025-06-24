""" test_yolo_training_program.py """

import unittest
import sys
import os
import shutil
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from yolo_training_program import YOLOTrainer, WholeImageDataset


class TestYOLOTrainer(unittest.TestCase):

    def setUp(self):
        # Create a temporary image directory with fake images
        self.test_dir = "temp_test_images"
        os.makedirs(self.test_dir, exist_ok=True)
        for i in range(5):
            img = Image.new("RGB", (512, 512), color=(255, 0, 0))
            img.save(os.path.join(self.test_dir, f"img_{i}.jpg"))


    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)


    def test_dataset_loading(self):
        dataset = WholeImageDataset(self.test_dir, transform=ToTensor())
        self.assertEqual(len(dataset), 5)
        img, label = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(label.shape, (1, 5))  # [cls, x_center, y_center, width, height]


    @patch("yolo_training_program.YOLO")
    def test_trainer_initialization(self, mock_yolo):
        # Mock model and parameters for optimizer
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_yolo.return_value.model = mock_model

        trainer = YOLOTrainer(self.test_dir, epochs=2, batch_size=2, img_size=256)
        self.assertEqual(trainer.epochs, 2)
        self.assertEqual(trainer.batch_size, 2)
        self.assertIsInstance(trainer.dataloader, DataLoader)
        # Check optimizer is created without error
        self.assertIsNotNone(trainer.optimizer)


    def test_iou_computation(self):
        boxA = [50, 50, 150, 150]
        boxB = [100, 100, 200, 200]
        iou = YOLOTrainer.compute_iou(boxA, boxB)
        self.assertTrue(0.0 < iou < 1.0)


    @patch("yolo_training_program.YOLO")
    def test_evaluate_accuracy_no_preds(self, mock_yolo):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, images, augment=False):
                batch_size = images.size(0)
                return [[torch.empty((0, 6)) for _ in range(batch_size)]]

            def eval(self):
                pass

        dummy_model = DummyModel()
        dummy_model.eval = MagicMock()

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.model = dummy_model
        mock_yolo.return_value = mock_yolo_instance

        trainer = YOLOTrainer(self.test_dir, epochs=1, batch_size=1, img_size=128)
        acc = trainer.evaluate_accuracy()
        self.assertEqual(acc, 0.0)


if __name__ == '__main__':
    unittest.main()
