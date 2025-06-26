""" beetle_cropper.py """

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
from globals import yolo_model, cropped_dataset

class BeetleCropper:
    """
    Uses a YOLOv8n model to crop beetles from images in a directory and saves the results to a new directory.
    """

    def __init__(self, image_dir, output_dir=cropped_dataset):
        """
        Constructor for the BeetleCropper class.

        Args:
            image_dir (str or Path): Path to directory containing input images.
            output_dir (str or Path): Path to directory where cropped images will be saved.
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        # initialize the yolov8n model
        self.yolo_model = YOLO(yolo_model)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

    def build(self):
        """
        Detects and crops beetles from images using YOLO. Cropped images are saved with the same filenames.
        """
        print(f"Cropping beetles from images in {self.image_dir}...")

        dropped_count = 0
        for img_file in self.image_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                img = Image.open(img_file).convert("RGB")
                img_array = np.array(img)

                results = self.yolo_model(img_array, imgsz=512)
                boxes = results[0].boxes

                if len(boxes) == 0:
                    dropped_count += 1
                    continue

                # Crop largest detected box
                largest_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                xyxy = largest_box.xyxy.cpu().numpy()[0].astype(int).tolist()
                cropped = img.crop(xyxy)

                # Save cropped image with original filename
                cropped.save(self.output_dir / img_file.name, format="JPEG")
            except Exception as e:
                print(f"Failed to process {img_file.name}: {e}")

        print(f"Dropped {dropped_count} image(s) from the dataset.")
        print(f"Cropped dataset saved to {self.output_dir}")

    def cleanup(self):
        """
        Deletes the cropped dataset directory and its contents.
        """
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            print(f"Cleaned up: {self.output_dir}")
