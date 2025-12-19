""" genus_specific_simulator.py """
import sys
import os
from PIL import Image
from beetle_cropper import BeetleCropper
from data_augmenter import DataAugmenter
from training_database_creator import TrainingDataConverter
from training_database_reader import DatabaseReader
from genus_specific_model_trainer import GenusSpecificModelTrainer
from model_loader import ModelLoader
import globals
from eval_species_by_genus import EvalSpeciesByGenus

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

if __name__ == '__main__':
    while True:
        print("\nWould you like to augment the dataset?")
        user_input = int(input("Enter 1 for YES, and 2 for NO: "))
        if user_input == 1:
            augment = True
            break
        if user_input == 2:
            augment = False
            break
        print("Invalid Input. Please enter 1 or 2.")

    while True:
        print("\nWould you like to run hyperparameter tuning or normal training?")
        user_input = int(input("Enter 1 for training, and 2 for tuning: "))
        if user_input == 1:
            train = True
            break
        if user_input == 2:
            train = False
            break
        print("Invalid input. Please enter a 1 or 2.")

    # Create the beetle cropper object to be used in dataset creation and image cropping
    beetle_cropper = BeetleCropper()
    # Crop the images in the original dataset so that the image is only the beetle
    beetle_cropper.build(image_dir="dataset", output_dir=globals.cropped_dataset)

    tdc = TrainingDataConverter(globals.cropped_dataset)
    tdc.conversion(globals.training_database)

    # Final cleanup: remove cropped dataset
    beetle_cropper.cleanup(globals.cropped_dataset)

    dbr = DatabaseReader(globals.training_database, class_file_path=globals.class_list)
    df = dbr.get_dataframe()

    genus_list = df['Genus'].unique().tolist()
    genus_specific_tp = GenusSpecificModelTrainer(dataframe=df, augment=augment)
    if train:
        print(f"Training for each genus in this list: {genus_list}")
        for genus in genus_list:
            genus_specific_tp.train_genus(genus, 20)

    else:
        print(f"Running hyperparameter tuning for each genus in this list: {genus_list}")
        for genus in genus_list:
            genus_specific_tp.run_optuna_study(genus, 20)

    # Load Genus models
    genus_model_paths = {
            "caud" : globals.gen_caud_model,
            "dors" : globals.gen_dors_model,
            "fron" : globals.gen_fron_model,
            "late" : globals.gen_late_model
        }

    GENUS_OUTPUTS = dbr.get_num_genus()
    genus_ml = ModelLoader(genus_model_paths, architecture="resnet50", num_classes=GENUS_OUTPUTS)
    genus_models = genus_ml.get_models()

    pic_evaluator = EvalSpeciesByGenus(genus_models, globals.gen_class_dictionary)

    # Get the images to be evaluated through user input

    DORS_PATH = "dataset/Algarobius prosopis GEM_3224221 5XEXT DORS.jpg"

    # Load the provided images
    DORS_IMG = Image.open(DORS_PATH) if DORS_PATH else None

    top_genus, top_species = pic_evaluator.classify_images(
        dors=DORS_IMG
    )

    print(f"Predicted Genus: {top_genus[0]}, Confidence: {top_genus[1]:.2f}\n")
    print(f"1. Predicted Species: {top_species[0][0]}, Confidence: {top_species[0][1]:.2f}\n")
    print(f"2. Predicted Species: {top_species[1][0]}, Confidence: {top_species[1][1]:.2f}\n")
    print(f"3. Predicted Species: {top_species[2][0]}, Confidence: {top_species[2][1]:.2f}\n")
    print(f"4. Predicted Species: {top_species[3][0]}, Confidence: {top_species[3][1]:.2f}\n")
    print(f"5. Predicted Species: {top_species[4][0]}, Confidence: {top_species[4][1]:.2f}\n")
