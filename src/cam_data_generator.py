""" cam_data_generator.py """

import os
from pathlib import Path
from PIL import Image
import dill
import torch
import torchvision.utils as vutils
import pandas as pd
import globals

from beetle_cropper import BeetleCropper
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader

class CAMDataGenerator:
    """
    Generates and saves transformed images per view using saved transformation files
    and matches original images by SpecimenID and View.
    """

    def __init__(self, dataframe, dataset_dir, image_column, class_column, class_string_dict, subsets, output_dir="cam_dataset"):
        """
        Args:
            dataframe (pd.DataFrame): Full dataset with 'SpecimenID' and 'View' columns.
            dataset_dir (str): Path to the original dataset directory containing .jpg files.
            image_column (str): Column name for image binaries (unused in this version but kept for compatibility).
            class_column (str): Column name for class labels.
            class_string_dict (dict): Mapping from label index to label string.
            subsets (dict): View-specific subsets of the dataframe.
            output_dir (str): Directory to save transformed images.
        """
        self.dataframe = dataframe
        self.dataset_dir = Path(dataset_dir)
        self.image_column = image_column
        self.class_column = class_column
        self.class_string_dict = class_string_dict
        self.subsets = subsets
        self.output_dir = Path(output_dir)

        self.transformation_paths = {
            "caud": globals.caud_transformation,
            "dors": globals.dors_transformation,
            "fron": globals.fron_transformation,
            "late": globals.late_transformation
        }

        self.transformations = self.load_transformations()

    def load_transformations(self):
        """
        Load transformation objects from .pth files using dill.

        Returns:
            dict: Dictionary mapping view name to transformation.
        """
        transformations = {}
        for view, path in self.transformation_paths.items():
            with open(path, "rb") as f:
                transformations[view] = dill.load(f)
        return transformations

    def find_image_path(self, specimen_id, view):
        """
        Searches the dataset directory for the exact image file using specimen ID and view.

        Args:
            specimen_id (str): Specimen ID string (e.g. '3216679').
            view (str): View string (e.g. 'DORS').

        Returns:
            Path or None: Path to the matched image file if found.
        """
        pattern = f"*{specimen_id}*{view}.jpg"
        matches = list(self.dataset_dir.rglob(pattern))
        return matches[0] if matches else None

    def save_transformed_images(self, samples_per_view=100):
        """
        Applies transformations and saves images for each view.

        Args:
            samples_per_view (int): Number of images to save per view.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for view, df in self.subsets.items():
            if df.empty:
                print(f"Skipping {view.upper()} — no data.")
                continue

            print(f"[{view.upper()}] Generating {samples_per_view} transformed images...")

            sample_df = df.sample(n=samples_per_view)
            labels = sample_df[self.class_column].values
            specimen_ids = sample_df["SpecimenID"].astype(str).values
            view_names = sample_df["View"].values

            transform = self.transformations[view]
            view_dir = self.output_dir / view
            view_dir.mkdir(parents=True, exist_ok=True)

            for i, (specimen_id, label, view_str) in enumerate(zip(specimen_ids, labels, view_names)):
                image_path = self.find_image_path(specimen_id, view_str)
                if image_path is None:
                    print(f"Could not find image for SpecimenID={specimen_id}, View={view_str}")
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"Failed to load image {image_path}: {e}")
                    continue

                transformed = transform(image)
                label_str = self.class_string_dict[label]
                filename = image_path.stem + f"_{label_str}.png"
                vutils.save_image(transformed, view_dir / filename)

            print(f"[{view.upper()}] Done. Saved to {view_dir}")

if __name__ == "__main__":

    # Create the beetle cropper object to be used in dataset creation and image cropping
    beetle_cropper = BeetleCropper()
    # Crop the images in the original dataset so that the image is only the beetle
    beetle_cropper.build(image_dir="dataset", output_dir=globals.cropped_dataset)

    # Set up data converter
    tdc = TrainingDataConverter(globals.cropped_dataset)
    tdc.conversion(globals.training_database)

    # Final cleanup: remove cropped dataset
    beetle_cropper.cleanup(globals.cropped_dataset)

    # Read converted data
    dbr = DatabaseReader(
        database=globals.training_database, class_file_path=globals.class_list)
    df = dbr.get_dataframe()

    subsets = {
        "caud" : df['View'] == "CAUD",
        "dors" : df['View'] == "DORS",
        "fron" : df['View'] == "FRON",
        "late" : df['View'] == "LATE"
    }

    generator = CAMDataGenerator(
        dataframe=df,
        dataset_dir="/dataset",
        image_column="Image",
        class_column="Species",
        class_string_dict=globals.spec_class_dictionary,
        subsets=subsets,
        output_dir="saved_augmented_images"
    )

generator.save_transformed_images(samples_per_view=100)
