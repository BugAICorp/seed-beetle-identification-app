""" dataframe_analyzer.py """

import sys
import os
import pandas as pd
from beetle_cropper import BeetleCropper
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

if __name__ == '__main__':

    while True:
        print("\nSelect which DataFrame you would like to analyze: ")
        print("\t1 = The Entire Dataset\n" \
            "\t2 = Dataset Limited to Class List\n" \
            "\t3 = The Entire Dataset Excluding the Class List")
        user_input = int(input("Enter the number of your choice: "))
        if user_input == 1:
            class_file_path = None
            exclude_classes = False
            break
        if user_input == 2:
            class_file_path = globals.class_list
            exclude_classes = False
            break
        if user_input == 3:
            class_file_path = globals.class_list
            exclude_classes = True
            break
        print("Invalid Input. Please enter 1, 2, or 3.")

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
        database=globals.training_database, class_file_path=class_file_path, exclude_classes=exclude_classes)
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

    output_dir = "dataframe_analysis"
    os.makedirs(output_dir, exist_ok=True)

    for v in views:
        genus_path = os.path.join(output_dir, f"genus_counts_{v}.csv")
        species_path = os.path.join(output_dir, f"species_counts_{v}.csv")

        # Save to CSV
        genus_tables[v].to_csv(genus_path, index=False)
        species_tables[v].to_csv(species_path, index=False)

        print(f"Saved genus counts for {v} view -> {genus_path}")
        print(f"Saved species counts for {v} view -> {species_path}")
