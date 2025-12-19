""" ood_simulator.py """

import sys
import os
import dill
from beetle_cropper import BeetleCropper
from training_database_creator import TrainingDataConverter
from training_database_reader import DatabaseReader
from model_loader import ModelLoader
from data_augmenter import DataAugmenter
from ood_tester import OODTester
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

if __name__ == '__main__':
    # Create the beetle cropper object to be used in dataset creation and image cropping
    beetle_cropper = BeetleCropper()
    # Crop the images in the original dataset so that the image is only the beetle
    beetle_cropper.build(image_dir="dataset", output_dir=globals.cropped_dataset)

    # Set up data converter
    tdc = TrainingDataConverter(globals.cropped_dataset)
    tdc.conversion(globals.training_database)

    # Final cleanup: remove cropped dataset
    beetle_cropper.cleanup(globals.cropped_dataset)

    # Get the id converted data using the class list and exclude_classes = False
    id_dbr = DatabaseReader(
        database=globals.training_database, class_file_path=globals.class_list, exclude_classes = False)

    id_df = id_dbr.get_dataframe()

    # Data Augmentation - Add images for rare classes
    id_augmenter = DataAugmenter(id_df, class_column="Species", threshold=100)

    id_df = id_augmenter.augment_rare_classes(num_augments_per_image=5)

    # Get the id converted data using the class list and exclude_classes = True
    ood_dbr = DatabaseReader(
        database=globals.training_database, class_file_path=globals.class_list, exclude_classes = True)

    ood_df = ood_dbr.get_dataframe()

    # Data Augmentation - Add images for rare classes
    ood_augmenter = DataAugmenter(ood_df, class_column="Species", threshold=100)

    ood_df = ood_augmenter.augment_rare_classes(num_augments_per_image=5)

    # Get the number of outputs
    SPECIES_OUTPUTS = id_dbr.get_num_species()
    GENUS_OUTPUTS = id_dbr.get_num_genus()

    # Setup model dictionaries
    species_model_filenames = {
        "caud" : globals.spec_caud_model, 
        "dors" : globals.spec_dors_model,
        "fron" : globals.spec_fron_model,
        "late" : globals.spec_late_model
    }

    genus_model_filenames = {
        "caud" : globals.gen_caud_model, 
        "dors" : globals.gen_dors_model,
        "fron" : globals.gen_fron_model,
        "late" : globals.gen_late_model
    }

    # Load the Models
    species_ml = ModelLoader(species_model_filenames, architecture="resnet50", num_classes=SPECIES_OUTPUTS)
    species_models = species_ml.get_models()

    genus_ml = ModelLoader(genus_model_filenames, architecture="resnet50", num_classes=GENUS_OUTPUTS)
    genus_models = genus_ml.get_models()

    # Get the transformations
    transformations = {}

    with open(globals.caud_transformation, "rb") as f:
        transformations["caud"] = dill.load(f)

    with open(globals.dors_transformation, "rb") as f:
        transformations["dors"] = dill.load(f)

    with open(globals.fron_transformation, "rb") as f:
        transformations["fron"] = dill.load(f)

    with open(globals.late_transformation, "rb") as f:
        transformations["late"] = dill.load(f)


    # Run Experiment
    temperatures = [1.0, 2.0, 5.0, 10.0]

    for key, model in species_models.items():
        species_tester = OODTester(
            model = model, id_dataframe = id_df, ood_dataframe = ood_df, transform = transformations[key])

        best_temp, results = species_tester.test_ood(temperatures = temperatures)
        species_tester.plot_distributions(
            results[best_temp]['id_energies'],
            results[best_temp]['ood_energies'],
            best_temp,
            output_dir = f"plots/{key}/species"
        )

    for key, model in genus_models.items():
        genus_tester = OODTester(
            model = model, id_dataframe = id_df, ood_dataframe = ood_df, transform = transformations[key])

        best_temp, results = genus_tester.test_ood(temperatures = temperatures)
        genus_tester.plot_distributions(
            results[best_temp]['id_energies'],
            results[best_temp]['ood_energies'],
            best_temp,
            output_dir = f"plots/{key}/genus"
        )
