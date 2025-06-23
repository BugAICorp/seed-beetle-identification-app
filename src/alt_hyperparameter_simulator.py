""" alt_hyperparameter_simulator.py """
import sys
import os
from PIL import Image
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from alt_training_program import AltTrainingProgram
from data_augmenter import DataAugmenter
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# simple simulation of end-to-end functionality of files

if __name__ == '__main__':
    train_dors_caud = False
    train_all = False
    train_dors_late = False
    can_continue = False

    while not can_continue:
        print("Dorsal-Caudal: 1\nAll: 2\nDorsal-Lateral: 3\nAll Models: 4")
        input = int(input(
            "Which model(s) would you like to run hyperparameter tuning on (type corresponding number): "))
        if input == 1:
            train_dors_caud = True
        elif input == 2:
            train_all = True
        elif input == 3:
            train_dors_late = True
        elif input == 4:
            train_dors_caud = True
            train_all = True
            train_dors_late = True
        else:
            print("Invalid Input")
        del input
        continue_input = int(
            input(
                "Press 1 to choose more models to train, anything other number to continue: "
                )
                )
        if continue_input != 1:
            can_continue = True
            if not train_dors_caud and not train_all and not train_dors_late:
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
        augmenter = DataAugmenter(df, class_column="Species", threshold=100)

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
    species_tp = AltTrainingProgram(dataframe=df, class_column="Species", num_classes=SPECIES_OUTPUTS)

    # Create dictionary to store best params for species models
    best_params_species = {}

    # Species hyperparameter tuning
    if train_dors_caud:
        best_params_species["dors_caud"] = species_tp.run_optuna_study(view="dors_caud")
    if train_all:
        best_params_species["all"] = species_tp.run_optuna_study(view="all")
    if train_dors_late:
        best_params_species["dors_late"] = species_tp.run_optuna_study(view="dors_late")

    # Run training with dataframe
    genus_tp = AltTrainingProgram(dataframe=df, class_column="Genus", num_classes=GENUS_OUTPUTS)

    # Create dictionary to store best params for genus models
    best_params_genus = {}

    # Genus hyperparameter tuning
    if train_dors_caud:
        best_params_genus["dors_aud"] = genus_tp.run_optuna_study(view="dors_caud")
    if train_all:
        best_params_genus["all"] = genus_tp.run_optuna_study(view="all")
    if train_dors_late:
        best_params_genus["dors_late"] = genus_tp.run_optuna_study(view="dors_late")

    # Print summary at the end
    print("\nSummary of Best Hyperparameters:\n")

    print("Species Model(s):")
    for view, params in best_params_species.items():
        print(f"  {view}: {params}")

    print("\nGenus Model(s):")
    for view, params in best_params_genus.items():
        print(f"  {view}: {params}")
