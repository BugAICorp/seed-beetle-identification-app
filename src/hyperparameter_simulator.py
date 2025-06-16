""" hyperparameter_simulator.py """
import sys
import os
from PIL import Image
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram
from data_augmenter import DataAugmenter
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# simple simulation of end-to-end functionality of files

if __name__ == '__main__':
    train_dors = False
    train_caud = False
    train_fron = False
    train_late = False
    can_continue = False

    while not can_continue:
        print("Dorsal: 1\nCaudal: 2\nFrontal: 3\nLateral: 4\nAll Models: 5")
        input = int(input(
            "Which model(s) would you like to run hyperparameter tuning on (type corresponding number): "))
        if input == 1:
            train_dors = True
        elif input == 2:
            train_caud = True
        elif input == 3:
            train_fron = True
        elif input == 4:
            train_late = True
        elif input == 5:
            train_dors = True
            train_caud = True
            train_fron = True
            train_late = True
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
            if not train_dors and not train_late and not train_caud and not train_fron:
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
    species_tp = TrainingProgram(df, "Species", SPECIES_OUTPUTS)

    # Create dictionary to store best params for species models
    best_params_species = {}

    # Species hyperparameter tuning
    if train_caud:
        best_params_species["caud"] = species_tp.run_optuna_study(view="caud")
    if train_dors:
        best_params_species["dors"] = species_tp.run_optuna_study(view="dors")
    if train_fron:
        best_params_species["fron"] = species_tp.run_optuna_study(view="fron")
    if train_late:
        best_params_species["late"] = species_tp.run_optuna_study(view="late")

    # Run training with dataframe
    genus_tp = TrainingProgram(df, "Genus", GENUS_OUTPUTS)

    # Create dictionary to store best params for genus models
    best_params_genus = {}

    # Genus hyperparameter tuning
    if train_caud:
        best_params_genus["caud"] = genus_tp.run_optuna_study(view="caud")
    if train_dors:
        best_params_genus["dors"] = genus_tp.run_optuna_study(view="dors")
    if train_fron:
        best_params_genus["fron"] = genus_tp.run_optuna_study(view="fron")
    if train_late:
        best_params_genus["late"] = genus_tp.run_optuna_study(view="late")

    # Print summary at the end
    print("\nSummary of Best Hyperparameters:\n")

    print("Species Model(s):")
    for view, params in best_params_species.items():
        print(f"  {view}: {params}")

    print("\nGenus Model(s):")
    for view, params in best_params_genus.items():
        print(f"  {view}: {params}")
