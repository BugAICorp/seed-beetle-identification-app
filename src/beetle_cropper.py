""" beetle_cropper.py """

import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
from globals import yolo_model


class BeetleCropper:
    """
    Uses a YOLOv8 model to crop beetles from images in a directory or a single image.
    """

    def __init__(self):
        """
        Initialize the YOLO model.
        """
        self.yolo_model = YOLO(yolo_model)

    def build(self, image_dir, output_dir):
        """
        Detects and crops beetles from all images in image_dir.
        Cropped images are saved with the same filenames to output_dir.

        Args:
            image_dir (str or Path): Directory containing input images.
            output_dir (str or Path): Directory where cropped images will be saved.
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        print(f"Cropping beetles from images in {image_dir}...")

        dropped_count = 0
        for img_file in image_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                img = Image.open(img_file).convert("RGB")
                cropped = self.crop_beetle(img)

                if cropped is None:
                    dropped_count += 1
                    continue

                cropped.save(output_dir / img_file.name, format="JPEG")

            except Exception as e:
                print(f"Failed to process {img_file.name}: {e}")

        print(f"Dropped {dropped_count} image(s) from the dataset.")
        print(f"Cropped dataset saved to {output_dir}")

    def crop_beetle(self, image: Image.Image):
        """
        Core YOLO crop logic used to crop the beetle out of a given image.
        Additionally, this is used by build().

        Args:
            image (PIL.Image): A PIL image.

        Returns:
            PIL.Image or None: Cropped beetle image or None if no box found.
        """
        img_array = np.array(image.convert("RGB"))
        results = self.yolo_model(img_array, imgsz=512)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return None

        largest_box = max(
            boxes,
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
        )
        xyxy = largest_box.xyxy.cpu().numpy()[0].astype(int).tolist()
        return image.crop(xyxy)

    def cleanup(self, output_dir):
        """
        Deletes the given cropped dataset directory and its contents.

        Args:
            output_dir (str or Path): Directory to delete.
        """
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Cleaned up: {output_dir}")
