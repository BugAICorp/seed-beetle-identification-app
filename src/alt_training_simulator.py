""" alt_training_simulator.py """
import sys
import os
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from alt_training_program import AltTrainingProgram
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

# simple simulation of end-to-end functionality of files

if __name__ == '__main__':
    log_file = open("training_comparison_output.log", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    # Set up data converter
    tdc = TrainingDataConverter("dataset")
    tdc.conversion(globals.training_database)
    # Read converted data
    dbr = DatabaseReader(globals.training_database, class_file_path=globals.class_list)
    original_df = dbr.get_dataframe()

    # Display how many images we have for each angle
    print("Number of Images for Each Angle in the Original Dataset:")
    print(f"CAUD: {(original_df['View'] == 'CAUD').sum()}")
    print(f"DORS: {(original_df['View'] == 'DORS').sum()}")
    print(f"FRON: {(original_df['View'] == 'FRON').sum()}")
    print(f"LATE: {(original_df['View'] == 'LATE').sum()}")

    # Data Augmentation - Add images for rare classes
    augmenter = DataAugmenter(original_df, class_column="Species", threshold=50)

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
    alt_species_tp = AltTrainingProgram(dataframe=df, class_column="Species", num_classes=SPECIES_OUTPUTS)

    # Training
    alt_species_tp.alt_train_resnet_model(num_epochs=20, view="dors_caud")
    alt_species_tp.alt_train_resnet_model(num_epochs=20, view="all")
    alt_species_tp.alt_train_resnet_model(num_epochs=20, view="dors_late")

    # Save models
    alt_species_model_filenames = {
            "dors_caud" : globals.spec_dors_caud_model,
            "all" : globals.spec_all_model,
            "dors_late": globals.spec_dors_late_model
        }

    alt_species_tp.save_models(
        alt_species_model_filenames,
        globals.alt_img_height,
        globals.alt_spec_class_dictionary,
        globals.alt_spec_accuracy_list)

    # Run training with dataframe
    alt_genus_tp = AltTrainingProgram(dataframe=df, class_column="Genus", num_classes=GENUS_OUTPUTS)

    # Training
    alt_genus_tp.alt_train_resnet_model(num_epochs=20, view="dors_caud")
    alt_genus_tp.alt_train_resnet_model(num_epochs=20, view="all")
    alt_genus_tp.alt_train_resnet_model(num_epochs=20, view="dors_late")

    # Save models
    alt_genus_model_filenames = {
        "dors_caud" : globals.gen_dors_caud_model, 
        "all" : globals.gen_all_model,
        "dors_late" : globals.gen_dors_late_model
    }

    alt_genus_tp.save_models(
        alt_genus_model_filenames,
        globals.alt_img_height,
        globals.alt_gen_class_dictionary,
        globals.alt_gen_accuracy_list)

    log_file.close()
