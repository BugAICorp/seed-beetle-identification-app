""" cam_data_generator.py """

import os
from pathlib import Path
from io import BytesIO
from PIL import Image
import dill
import torch
import torchvision.utils as vutils
import globals

class CAMDataGenerator:
    """
    Generates and saves transformed images per view using saved transformation files.
    """

    def __init__(self, dataframe, image_column, class_column, class_string_dict, subsets, output_dir="transformed_dataset"):
        """
        Args:
            dataframe (pd.DataFrame): Full dataset.
            image_column (str): Column name for image binaries.
            class_column (str): Column name for class labels.
            class_string_dict (dict): Mapping from label index to label string.
            subsets (dict): View-specific subsets of the dataframe.
            output_dir (str): Base directory to save transformed images.
        """
        self.dataframe = dataframe
        self.image_column = image_column
        self.class_column = class_column
        self.class_string_dict = class_string_dict
        self.subsets = subsets
        self.transformation_paths = {
            "caud" : globals.caud_transformation,
            "dors" : globals.dors_transformation,
            "fron" : globals.fron_transformation,
            "late" : globals.late_transformation
        }
        self.output_dir = output_dir

        # Load transformations from .pth files
        self.transformations = self.load_transformations()

    def load_transformations(self):
        """
        Loads transformation objects from .pth files using dill.

        Returns:
            dict: Dictionary mapping view name to transformation.
        """
        transformations = {}
        for view, path in self.transformation_paths.items():
            with open(path, "rb") as f:
                transformations[view] = dill.load(f)
        return transformations

    def save_transformed_images(self, samples_per_view=100):
        """
        Applies transformations and saves images for each view.

        Args:
            samples_per_view (int): Number of images to save per view.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        for view, df in self.subsets.items():
            if df.empty:
                print(f"Skipping {view.upper()} — no data.")
                continue

            print(f"[{view.upper()}] Generating {samples_per_view} transformed images...")

            sample_df = df.sample(n=samples_per_view)
            image_binaries = sample_df[self.image_column].values
            labels = sample_df[self.class_column].values

            transform = self.transformations[view]

            view_dir = Path(self.output_dir) / view
            view_dir.mkdir(parents=True, exist_ok=True)

            for i, (binary, label) in enumerate(zip(image_binaries, labels)):
                image = Image.open(BytesIO(binary)).convert("RGB")
                transformed = transform(image)
                label_str = self.class_string_dict[label]
                filename = f"{view}_{i:03d}_{label_str}.png"
                vutils.save_image(transformed, view_dir / filename)

            print(f"[{view.upper()}] Done. Saved to {view_dir}")
