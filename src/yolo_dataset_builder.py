""" yolo_dataset_builder.py """

import os
import shutil
import random
from pathlib import Path
import yaml

class YoloDatasetBuilder:
    """
    Dataset Builder that creates directories to be used in the YOLOv8 training process.
    """
    def __init__(self, source_dir="dataset", output_dir="yolo_dataset", train_ratio=0.8):
        """
        DatasetBuilder constructor.

        Args:
            source_dir (str): Directory containing all source images.
            output_dir (str): Directory where the train/val split will be created.
            train_ratio (float): Proportion of images to use for training (e.g., 0.8 = 80% train, 20% val).
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.yaml = "yolo_dataset.yaml"
        self.train_ratio = train_ratio
        self.image_types = ['.jpg', '.jpeg', '.png']

    def build(self, total_images=None):
        """
        Builds the directory structure and copies images for YOLO training.

        Args:
            total_images (int): Maximum number of images to copy (total, across train + val).
        """
        image_paths = [p for p in self.source_dir.iterdir() if p.is_file() and p.suffix.lower() in self.image_types]

        # Subsample if max_images is set
        if total_images is not None:
            image_paths = image_paths[:total_images]

        random.shuffle(image_paths)

        split_idx = int(len(image_paths) * self.train_ratio)
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        self.copy_images(train_paths, 'train')
        self.copy_images(val_paths, 'val')
        self.create_yaml()

    def copy_images(self, images, split):
        """
        Copies images to the split directory.

        Args:
            images (List[Path]): List of image file paths to copy.
            split (str): Subdirectory name to copy images into ('train' or 'val').
        """
        img_dir = self.output_dir / split / 'images'
        lbl_dir = self.output_dir / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            shutil.copy(img, img_dir / img.name)

            # Create dummy label file (YOLO format: class x_center y_center width height)
            label_path = lbl_dir / img.with_suffix('.txt').name
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

    def create_yaml(self):
        """
        Creates the YAML file to be used during training.
        """
        data = {
            "train": str(self.output_dir / "train/images"),
            "val": str(self.output_dir / "val/images"),
            "nc": 1,
            "names": ["Seed Beetle"]
        }
        with open(self.yaml, 'w') as f:
            yaml.dump(data, f)

    def cleanup(self):
        """ Deletes all created directories and the YAML file. """
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            print(f"Removed dataset directory: {self.output_dir}")
        else:
            print(f"No dataset directory found at: {self.output_dir}")
        yaml_path = Path(self.yaml)
        if yaml_path.exists():
            yaml_path.unlink()
            print(f"Removed YAML file: {self.yaml}")
        else:
            print(f"No YAML file found at: {self.yaml}")
