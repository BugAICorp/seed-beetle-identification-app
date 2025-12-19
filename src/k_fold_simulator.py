""" k_fold_simulator.py """
import sys
import os
from training_database_creator import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram
from beetle_cropper import BeetleCropper
import globals
from import_hyperparams import import_params

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class Tee:
    """
    Class to enable stdout to output to both a log file and stdout in terminal
    """
    def __init__(self, *streams):
        """ Stores streams """
        self.streams = streams

    def write(self, message):
        """ Write to all output streams """
        for s in self.streams:
            s.write(message)
            s.flush()  # Ensure it gets written immediately

    def flush(self):
        """ Flush after write to avoid buffering """
        for s in self.streams:
            s.flush()

# Simple simulation of stratified k-fold validation for model testing
if __name__ == '__main__':
    log_file = open("stratified_k_fold_output.log", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    try:
        k_fold_dors = False
        k_fold_caud = False
        k_fold_fron = False
        k_fold_late = False
        can_continue = False

        while not can_continue:
            print("Dorsal: 1\nCaudal: 2\nFrontal: 3\nLateral: 4")
            input = int(input(
                "Choose a model you would like to run stratified k-fold validation on (type corresponding number): "))
            if input == 1:
                k_fold_dors = True
            elif input == 2:
                k_fold_caud = True
            elif input == 3:
                k_fold_fron = True
            elif input == 4:
                k_fold_late = True
            elif input == 5:
                k_fold_dors = True
                k_fold_caud = True
                k_fold_fron = True
                k_fold_late = True
            else:
                print("Invalid Input")
            del input
            continue_input = int(
                input(
                    "Press 1 to choose more models to train, anything other number to start training: "
                    )
                    )
            if continue_input != 1:
                can_continue = True
                if not k_fold_dors and not k_fold_late and not k_fold_caud and not k_fold_fron:
                    print("No Training Requested")
                    sys.exit(0)

        while True:
            print("\nWhich model architecture would you like to use?")
            user_input = int(input("Enter 1 for ResNet18, and 2 for ResNet50: "))
            if user_input == 1:
                architecture = "resnet18"
                break
            if user_input == 2:
                architecture = "resnet50"
                break
            print("Invalid Input. Please enter 1 or 2.")

        while True:
            print("\nWould you like to train with an \"other\" class?")
            user_input = int(input("Enter 1 for YES, and 2 for NO: "))
            # If yes, set model paths to the "other" paths
            if user_input == 1:
                # set create other flag to true
                create_other = True
                break
            # If no, set model paths to the normal paths
            if user_input == 2:
                # set create other flag to false
                create_other = False
                break
            print("Invalid Input. Please enter 1 or 2.")

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
            print("\nWould you like to use class balancing techniques while training?")
            print("\t0 = No Balancing\n" \
                "\t1 = Class-Weighted Loss Function\n" \
                "\t2 = Oversampling Only\n" \
                "\t3 = Both (Oversampling + Class-Weighted Loss)")
            user_input = int(input("Enter the number of your choice: "))
            if user_input == 0:
                balance_classes = 0
                break
            if user_input == 1:
                balance_classes = 1
                break
            if user_input == 2:
                balance_classes = 2
                break
            if user_input == 3:
                balance_classes = 3
                break
            print("Invalid Input. Please enter 0, 1, 2, or 3.")

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
            database=globals.training_database, class_file_path=globals.class_list, create_other=create_other)
        df = dbr.get_dataframe()

        # Display how many images we have for each angle
        print("Number of Images for Each Angle in the Original Dataset:")
        print(f"CAUD: {(df['View'] == 'CAUD').sum()}")
        print(f"DORS: {(df['View'] == 'DORS').sum()}")
        print(f"FRON: {(df['View'] == 'FRON').sum()}")
        print(f"LATE: {(df['View'] == 'LATE').sum()}")

        # Initialize number of outputs
        SPECIES_OUTPUTS = dbr.get_num_species()
        GENUS_OUTPUTS = dbr.get_num_genus()

        # Run training with dataframe
        species_tp = TrainingProgram(
            df, "Species", SPECIES_OUTPUTS, architecture=architecture, augment=augment, balance_classes=balance_classes
        )

        # Training
        if k_fold_caud:
            hyperparameters = import_params(globals.species_caud_hypers)
            species_tp.k_fold_resnet(k_folds=5, **hyperparameters)

        if k_fold_dors:
            hyperparameters = import_params(globals.species_dors_hypers)
            species_tp.k_fold_resnet(k_folds=5, **hyperparameters)

        if k_fold_fron:
            hyperparameters = import_params(globals.species_fron_hypers)
            species_tp.k_fold_resnet(k_folds=5, **hyperparameters)

        if k_fold_late:
            hyperparameters = import_params(globals.species_late_hypers)
            species_tp.k_fold_resnet(k_folds=5, **hyperparameters)

        # Run training with dataframe
        genus_tp = TrainingProgram(
            df, "Genus", GENUS_OUTPUTS, architecture=architecture, augment=augment, balance_classes=balance_classes
        )

        # Training
        if k_fold_caud:
            hyperparameters = import_params(globals.genus_caud_hypers)
            genus_tp.k_fold_resnet(k_folds=5, **hyperparameters)

        if k_fold_dors:
            hyperparameters = import_params(globals.genus_dors_hypers)
            genus_tp.k_fold_resnet(k_folds=5, **hyperparameters)

        if k_fold_fron:
            hyperparameters = import_params(globals.genus_fron_hypers)
            genus_tp.k_fold_resnet(k_folds=5, **hyperparameters)

        if k_fold_late:
            hyperparameters = import_params(globals.genus_late_hypers)
            genus_tp.k_fold_resnet(k_folds=5, **hyperparameters)

    finally:
        log_file.close()
