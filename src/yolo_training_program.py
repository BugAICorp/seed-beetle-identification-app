""" yolo_training_program.py """

import os
import glob
import shutil

from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])

from ultralytics import YOLO
import torch
from globals import yolo_model

_original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    """
    Patched version of torch.load that forces weights_only=False.

    This function overrides the default behavior of torch.load in PyTorch >=2.6,
    where weights_only=True is the new default. By explicitly setting 
    weights_only=False, it ensures that full model objects can be deserialized 
    properly.

    WARNING: Setting weights_only=False can execute arbitrary code during 
    unpickling. Only use this patch if the source of the checkpoint file is 
    fully trusted.

    Args:
        f (str or file-like): The file path or object from which to load the model.
        *args: Additional positional arguments to pass to torch.load.
        **kwargs: Additional keyword arguments to pass to torch.load.

    Returns:
        The deserialized object (a model or checkpoint dictionary).
    """
    kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

class YOLOTrainer:
    """
    YOLOv8 training class for whole image bounding box detection of a single class.
    """
    def __init__(
            self,
            dataset_yaml,
            complex_training=False,
            epochs=40,
            batch_size=8,
            img_size=512,
            device=None
    ):
        """
        Initializer for the YOLOTrainer class.

        Args:
            dataset_yaml (str): Path to YOLO dataset YAML.
            complex_training (bool): If True, use advanced augmentations and fine-tuning.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for DataLoader.
            img_size (int): Image resize size.
            device (torch.device): Device for training (auto-selected if None).
        """
        self.dataset_yaml = dataset_yaml
        self.complex_training = complex_training
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_built() else
            "cpu"
        )
        # Patch torch.load to force weights_only=False during load
        torch.load = patched_torch_load
        self.model = YOLO("yolov8n.pt")
        torch.load = _original_torch_load
        self.model.to(self.device)

    def train(self):
        """
        Runs training loop for the specified number of epochs on the YOLOv8n model.
        Uses enhanced training configuration if complex_training=True
        """
        print(f"Starting YOLOv8 training on {self.device}:")

        if self.complex_training:
            self.model.train(
                data=self.dataset_yaml,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.img_size,
                device=str(self.device),
                # Optimization settings
                lr0=0.0015,            # Lower LR for fine-tuning stability
                lrf=0.01,              # Final LR = Slow LR decay
                optimizer="AdamW",     # Better generalization for small datasets
                dropout=0.05,          # Prevent overfitting
                # Learning rate schedule
                cos_lr=True,           # Smooth cosine schedule
                # Loss function weights
                cls=2.0,               # Increase classification loss to reduce false positives
                box=1.2,               # Improve bounding box precision
                dfl=1.0,               # Distribution focal loss scaling
                # Color Augmentations
                hsv_h=0.015,           # Small hue jitter (preserves natural beetle colors)
                hsv_s=0.7,             # Larger saturation shift for robustness
                hsv_v=0.4,             # Moderate brightness/contrast variation
                # Geometric Augmentations
                translate=0.1,         # Moderate translation
                scale=0.6,             # Moderate zoom-in/out
                degrees=5,             # Small rotations
                fliplr=0.5,            # 50% probability of horizontal flip
                # Mosaic / multi-scale
                multi_scale=False,     # Disabled, improves scale robustness
                mosaic=0.5,            # Mosaic augmentation half the time (detection diversity)
                close_mosaic=10,       # Close before convergence
                # Regularization
                erasing=0.3,           # Random erasing to help with generalization
                # Training behavior
                patience=50,           # Early stop if no improvement
                augment=True,          # Enable YOLO’s built-in augmentations
                deterministic=True,    # Ensures deterministic behavior for reproducible training
            )
        else:
            # Basic training configuration
            self.model.train(
                data=self.dataset_yaml,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.img_size,
                device=str(self.device)
            )

    def save(self, save_path=yolo_model):
        """
        Automatically locate the most recent YOLO training run and copy the best model weights.
        """
        # Find all 'train*' folders inside runs/detect/
        run_dirs = glob.glob("runs/detect/train*")
        if not run_dirs:
            raise FileNotFoundError("No YOLO training runs found in 'runs/detect/'.")

        # Get the most recently modified run directory
        latest_run = max(run_dirs, key=os.path.getmtime)

        # Construct expected best weight path
        best_model_path = os.path.join(latest_run, "weights", "best.pt")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Could not find best.pt in {latest_run}")

        # Copy model to desired save path
        shutil.copy(best_model_path, save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    trainer = YOLOTrainer(
        dataset_yaml="yolo_dataset/data.yaml",
        complex_training=True
    )
    trainer.train()
    trainer.save()
