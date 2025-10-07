""" k_fold_simulator.py """
import sys
import os
from training_database_creator import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram
from beetle_cropper import BeetleCropper
import globals

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
        species_tp = TrainingProgram(df, "Species", SPECIES_OUTPUTS, augment=augment, balance_classes=balance_classes)

        # Training
        if k_fold_caud:
            erasure_params_caud = {
                "p": 0.3978357251429255,
                "min": 0.04237603082954706,
                "max": 0.3025963685284483
            }
            species_tp.k_fold_resnet(20, "caud", k_folds=5, batch=16, rotation=16,
                                     brightness=0.04160844, lrate=0.0002188637,
                                     erasure_params=erasure_params_caud)
        if k_fold_dors:
            erasure_params_dors = {
                "p": 0.5763301129483613,
                "min": 0.06044662804540117,
                "max": 0.18387577071515754
            }
            species_tp.k_fold_resnet(20, "dors", k_folds=5, batch=64, rotation=4,
                                     brightness=0.2320837289, lrate=0.00042698,
                                     erasure_params=erasure_params_dors)
        if k_fold_fron:
            erasure_params_fron = {
                "p": 0.265585095702728,
                "min": 0.071779115882381,
                "max": 0.29234187228616554
            }
            species_tp.k_fold_resnet(20, "fron", k_folds=5, batch=32, rotation=3,
                                     brightness=0.124352955, lrate=0.0002323599,
                                     erasure_params=erasure_params_fron)
        if k_fold_late:
            erasure_params_late = {
                "p": 0.5189325280363017,
                "min": 0.03843699036307908,
                "max": 0.11129682877722781
            }
            species_tp.k_fold_resnet(20, "late", k_folds=5, batch=32, rotation=16,
                                     brightness=0.05717608, lrate=0.00036962807,
                                     erasure_params=erasure_params_late)

        # Run training with dataframe
        genus_tp = TrainingProgram(df, "Genus", GENUS_OUTPUTS, augment=augment, balance_classes=balance_classes)

        # Training
        if k_fold_caud:
            erasure_params_caud = {
                "p": 0.3127187908868738,
                "min": 0.04046194894255532,
                "max": 0.29175754421281885
            }
            genus_tp.k_fold_resnet(20, "caud", k_folds=5, batch=16, rotation=2,
                                   brightness=0.121347939, lrate=0.000414240154,
                                   erasure_params=erasure_params_caud)
        if k_fold_dors:
            erasure_params_dors = {
                "p": 0.08429225010786912,
                "min": 0.05881609232667761,
                "max": 0.29034641815208423
            }
            genus_tp.k_fold_resnet(20, "dors", k_folds=5, batch=16, rotation=13,
                                   brightness=0.169855976, lrate=0.000179720464,
                                   erasure_params=erasure_params_dors)
        if k_fold_fron:
            erasure_params_fron = {
                "p": 0.7558902433519469,
                "min": 0.07276752102604624,
                "max": 0.1953562902391759
            }
            genus_tp.k_fold_resnet(20, "fron", k_folds=5, batch=16, rotation=6,
                                   brightness=0.05464547869, lrate=0.0002265474186,
                                   erasure_params=erasure_params_fron)
        if k_fold_late:
            erasure_params_late = {
                "p": 0.3860968267885073,
                "min": 0.09392431854817945,
                "max": 0.2564630945836204
            }
            genus_tp.k_fold_resnet(20, "late", k_folds=5, batch=32, rotation=10,
                                   brightness=0.29610847517, lrate=0.0001860446889,
                                   erasure_params=erasure_params_late)

    finally:
        log_file.close()
