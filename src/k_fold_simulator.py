""" k_fold_simulator.py """
import sys
import os
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram
from data_augmenter import DataAugmenter
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

# simple simulation of stratified k-fold validation for model testing
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
            print("\nWould you like to augment the dataset?")
            user_input = int(input("Enter 1 for YES, and 2 for NO: "))
            if user_input == 1:
                augment = True
                break
            if user_input == 2:
                augment = False
                break
            print("Invalid Input. Please enter 1 or 2.")

        # Set up data converter
        tdc = TrainingDataConverter("dataset")
        tdc.conversion(globals.training_database)
        # Read converted data
        dbr = DatabaseReader(database=globals.training_database, class_file_path=globals.class_list)
        df = dbr.get_dataframe()

        # Display how many images we have for each angle
        print("Number of Images for Each Angle in the Original Dataset:")
        print(f"CAUD: {(df['View'] == 'CAUD').sum()}")
        print(f"DORS: {(df['View'] == 'DORS').sum()}")
        print(f"FRON: {(df['View'] == 'FRON').sum()}")
        print(f"LATE: {(df['View'] == 'LATE').sum()}")

        if augment:
            # Data Augmentation - Add images for rare classes
            augmenter = DataAugmenter(df, class_column="Species", threshold=50)

            df = augmenter.augment_rare_classes(num_augments_per_image=5)

            # Display how many images we have for each angle after augmenting the data
            print("\nNumber of Images for Each Angle After Augmentation:")
            print(f"CAUD: {(df['View'] == 'CAUD').sum()}")
            print(f"DORS: {(df['View'] == 'DORS').sum()}")
            print(f"FRON: {(df['View'] == 'FRON').sum()}")
            print(f"LATE: {(df['View'] == 'LATE').sum()}")

        # initialize number of outputs
        SPECIES_OUTPUTS = dbr.get_num_species()
        GENUS_OUTPUTS = dbr.get_num_genus()

        # Run training with dataframe
        species_tp = TrainingProgram(df, "Species", SPECIES_OUTPUTS)

        # Training
        if k_fold_caud:
            species_tp.k_fold_resnet(20, "caud", k_folds=5)
        if k_fold_dors:
            species_tp.k_fold_resnet(20, "dors", k_folds=5)
        if k_fold_fron:
            species_tp.k_fold_resnet(20, "fron", k_folds=5)
        if k_fold_late:
            species_tp.k_fold_resnet(20, "late", k_folds=5)

        # Run training with dataframe
        genus_tp = TrainingProgram(df, "Genus", GENUS_OUTPUTS)

        # Training
        if k_fold_caud:
            genus_tp.k_fold_resnet(20, "caud", k_folds=5)
        if k_fold_dors:
            genus_tp.k_fold_resnet(20, "dors", k_folds=5)
        if k_fold_fron:
            genus_tp.k_fold_resnet(20, "fron", k_folds=5)
        if k_fold_late:
            genus_tp.k_fold_resnet(20, "late", k_folds=5)

    finally:
        log_file.close()
