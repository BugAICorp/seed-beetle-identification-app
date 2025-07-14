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

    def __init__(self, dataframe, dataset_dir, subsets, output_dir="cam_dataset"):
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
        self.image_column = "Image"
        self.class_column = "Genus"
        self.class_string_dict = globals.gen_class_dictionary
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
        pattern = f"*{specimen_id}*{view.upper()}.jpg"
        matches = list(self.dataset_dir.rglob(pattern))
        return matches[0] if matches else None

    def save_transformed_images(self, samples_per_view=100):
        """
        Applies transformations and saves images for each view,
        ensuring all genera are represented, with proportional remainder distribution.

        Args:
            samples_per_view (int): Total number of images to save per view.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for view, df in self.subsets.items():
            if df.empty:
                print(f"Skipping {view.upper()} — no data.")
                continue

            print(f"[{view.upper()}] Generating {samples_per_view} balanced transformed images...")

            # Map label index to genus string
            df["Genus"] = df[self.class_column].map(self.class_string_dict)
            genus_groups = df.groupby("Genus")
            num_genera = len(genus_groups)

            if num_genera == 0:
                print(f"No genera found in {view.upper()} view.")
                continue

            # Count how many samples each genus has
            genus_sizes = genus_groups.size().to_dict()
            total_available = sum(genus_sizes.values())

            # Compute fair allocation: ensure every genus gets at least one, then proportional remainder
            base_allocation = {genus: 1 for genus in genus_sizes}
            remaining = samples_per_view - num_genera

            if remaining > 0:
                total_weights = total_available - num_genera
                # This gives more samples to larger genera, proportionally
                for genus, size in genus_sizes.items():
                    weight = size - 1
                    if weight <= 0:
                        continue
                    addl = round((weight / total_weights) * remaining)
                    base_allocation[genus] += addl

            # Sample from each genus
            balanced_samples = []
            for genus, group in genus_groups:
                n_samples = min(base_allocation[genus], len(group))
                sampled = group.sample(n=n_samples, random_state=7)
                balanced_samples.append(sampled)

            balanced_df = pd.concat(balanced_samples).reset_index(drop=True)
            transform = self.transformations[view]
            view_dir = self.output_dir / view
            view_dir.mkdir(parents=True, exist_ok=True)

            for _, row in balanced_df.iterrows():
                specimen_id = str(row["SpecimenID"])
                label_idx = row[self.class_column]
                view_str = row["View"]

                image_path = self.find_image_path(specimen_id, view_str)
                if image_path is None:
                    print(f"Could not find image for SpecimenID={specimen_id}, View={view_str}")
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")
                except OSError:
                    print(f"Failed to load image: {image_path}")
                    continue

                transformed = transform(image)
                filename = image_path.name  # preserve original filename
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
        database=globals.training_database,
        class_file_path=globals.class_list
    )
    dataframe = dbr.get_dataframe()

    view_subsets = {
        "caud": dataframe[dataframe['View'] == "CAUD"],
        "dors": dataframe[dataframe['View'] == "DORS"],
        "fron": dataframe[dataframe['View'] == "FRON"],
        "late": dataframe[dataframe['View'] == "LATE"]
    }

    generator = CAMDataGenerator( # pylint: disable=possibly-used-before-assignment
        dataframe=dataframe,
        dataset_dir="dataset",
        subsets=view_subsets,
        output_dir="cam_dataset"
    )

    generator.save_transformed_images(samples_per_view=100)
