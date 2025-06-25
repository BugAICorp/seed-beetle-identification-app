""" yolo_training_program.py """

from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])
from ultralytics import YOLO
import torch
from globals import yolo_model

class YOLOTrainer:
    """
    YOLOv8 training class for whole image bounding box detection of a single class.

    Args:
        dataset_path (str): Directory with all images.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        img_size (int): Image resize size.
        device (torch.device): Device for training (auto-selected if None).
    """
    def __init__(self, dataset_yaml, epochs=10, batch_size=8, img_size=512, device=None):
        self.dataset_yaml = dataset_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_built() else
            "cpu"
        )
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)

    def train(self):
        """
        Runs training loop for the specified number of epochs on the YOLOv8n model.
        """
        # Call Ultralytics built-in train method
        self.model.train(
            data=self.dataset_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            device=str(self.device)
        )

    def save(self, save_path=yolo_model):
        """
        Save trained model weights.
        """
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    trainer = YOLOTrainer(dataset_yaml="dataset.yaml")
    trainer.train()
    trainer.save()
