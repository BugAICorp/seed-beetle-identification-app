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
            erasure_params_caud = {
                "p": 0.5450068594306283,
                "min": 0.032231275920186486,
                "max": 0.23975077356392424
            }
            species_tp.k_fold_resnet(20, "caud", k_folds=5, batch=16, rotation=6,
                                     brightness=0.0672682540489113, lrate=0.0002205207835665262,
                                     erasure_params=erasure_params_caud, max_os_ratio=1.5)
        if k_fold_dors:
            erasure_params_dors = {
                "p": 0.7757711509313643,
                "min": 0.01008374654178916,
                "max": 0.38794012670750844
            }
            species_tp.k_fold_resnet(20, "dors", k_folds=5, batch=16, rotation=12,
                                     brightness=0.22216817398095146, lrate=0.0001296278789334687,
                                     erasure_params=erasure_params_dors, max_os_ratio=2.5)
        if k_fold_fron:
            erasure_params_fron = {
                "p": 0.14786083200104405,
                "min": 0.08542272176573411,
                "max": 0.3766890143419105
            }
            species_tp.k_fold_resnet(20, "fron", k_folds=5, batch=16, rotation=7,
                                     brightness=0.16052298566019538, lrate=0.00018151090290770348,
                                     erasure_params=erasure_params_fron, max_os_ratio=4.0)
        if k_fold_late:
            erasure_params_late = {
                "p": 0.005799105801707227,
                "min": 0.08818090418966613,
                "max": 0.2566152645216
            }
            species_tp.k_fold_resnet(20, "late", k_folds=5, batch=64, rotation=6,
                                     brightness=0.29977566775503983, lrate=0.00012089084719947084,
                                     erasure_params=erasure_params_late, max_os_ratio=3.5)

        # Run training with dataframe
        genus_tp = TrainingProgram(
            df, "Genus", GENUS_OUTPUTS, architecture=architecture, augment=augment, balance_classes=balance_classes
        )

        # Training
        if k_fold_caud:
            erasure_params_caud = {
                "p": 0.117534992000064,
                "min": 0.08054270560117567,
                "max": 0.2983577819330524
            }
            genus_tp.k_fold_resnet(20, "caud", k_folds=5, batch=16, rotation=10,
                                   brightness=0.1462847736327197, lrate=0.00004409398823911199,
                                   erasure_params=erasure_params_caud, max_os_ratio=5.0)
        if k_fold_dors:
            erasure_params_dors = {
                "p": 0.6279748323341047,
                "min": 0.041921505805665914,
                "max": 0.24388226488220693
            }
            genus_tp.k_fold_resnet(20, "dors", k_folds=5, batch=32, rotation=6,
                                   brightness=0.2988104061389692, lrate=0.00004736821824349854,
                                   erasure_params=erasure_params_dors, max_os_ratio=1.0)
        if k_fold_fron:
            erasure_params_fron = {
                "p": 0.30518586009082976,
                "min": 0.04609315007975057,
                "max": 0.36140797065499464
            }
            genus_tp.k_fold_resnet(20, "fron", k_folds=5, batch=64, rotation=14,
                                   brightness=0.22903306674663448, lrate=0.0001380146193447115,
                                   erasure_params=erasure_params_fron, max_os_ratio=5.0)
        if k_fold_late:
            erasure_params_late = {
                "p": 0.30535724516213314,
                "min": 0.011359991265195598,
                "max": 0.31162030351760406
            }
            genus_tp.k_fold_resnet(20, "late", k_folds=5, batch=32, rotation=10,
                                   brightness=0.04304050259182124, lrate=0.00001826137626671228,
                                   erasure_params=erasure_params_late, max_os_ratio=3.0)

    finally:
        log_file.close()
