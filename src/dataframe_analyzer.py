""" dataframe_analyzer.py """

import sys
import os
from beetle_cropper import BeetleCropper
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

if __name__ == '__main__':

    while True:
        choice = input("\nWould you like to limit the DataFrame based on the class list? (y/n): ").lower()
        if choice == 'y':
            class_file_path = globals.class_list
            break
        if choice == 'n':
            class_file_path = None
            break
        print("Invalid input. Please try again.")

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
        database=globals.training_database, class_file_path=class_file_path)
    df = dbr.get_dataframe()

    # Display how many images we have for each angle
    print("Number of Images for Each Angle in the Original Dataset:")
    print(f"CAUD: {(df['View'] == 'CAUD').sum()}")
    print(f"DORS: {(df['View'] == 'DORS').sum()}")
    print(f"FRON: {(df['View'] == 'FRON').sum()}")
    print(f"LATE: {(df['View'] == 'LATE').sum()}")

    views = ["DORS", "LATE", "FRON", "CAUD"]

    # Genus-level counts per view
    genus_tables = {}
    for v in views:
        genus_tables[v] = (
            df[df["View"] == v]
            .groupby("Genus")["Filename"]
            .count()
            .reset_index(name="ImageCount")
            .sort_values("ImageCount", ascending=False)
        )

    # Species-level counts per view
    species_tables = {}
    for v in views:
        species_tables[v] = (
            df[df["View"] == v]
            .groupby(["Genus", "Species"])["Filename"]
            .count()
            .reset_index(name="ImageCount")
            .sort_values("ImageCount", ascending=False)
        )

    for v in views:
        print(f"Genus counts - {v} view:")
        print(genus_tables[v])

        print(f"\nSpecies counts - {v} view:")
        print(species_tables[v])
