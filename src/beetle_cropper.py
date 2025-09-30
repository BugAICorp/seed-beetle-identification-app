""" beetle_cropper.py """

import shutil
from pathlib import Path

from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])

from ultralytics import YOLO
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
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

class BeetleCropper:
    """
    Uses a YOLOv8 model to crop beetles from images in a directory or a single image.
    """

    def __init__(self, threshold=0.25):
        """
        Initialize the YOLO model.

        Args:
            threshold (float): Minimum confidence score required for a detection 
                to be considered valid. Detections below this threshold will be 
                ignored, and the image will be rejected if no boxes meet the 
                threshold.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_built() else
            "cpu"
        )
        torch.load = patched_torch_load
        self.yolo_model = YOLO(yolo_model)
        torch.load = _original_torch_load
        self.yolo_model.to(self.device)

        self.threshold = threshold

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

        dropped_files = []
        for img_file in image_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                img = Image.open(img_file).convert("RGB")
                cropped = self.crop_beetle(img)

                if cropped is None:
                    dropped_files.append(img_file.name)
                    continue

                cropped.save(output_dir / img_file.name, format="JPEG")

            except UnidentifiedImageError:
                print(f"Cannot identify image file: {img_file.name}")
                dropped_files.append(img_file.name)
            except OSError as e:
                print(f"OS error processing {img_file.name}: {e}")
                dropped_files.append(img_file.name)

        print(f"Dropped {len(dropped_files)} image(s) from the dataset.")
        if dropped_files:
            print("Dropped files:")
            for f in dropped_files:
                print("  -", f)
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
        results = self.yolo_model(img_array, imgsz=512, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return None

        # Filter by confidence
        boxes = [b for b in boxes if b.conf.cpu().item() >= self.threshold]
        if len(boxes) == 0:
            return None

        # Choose largest surviving box
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
