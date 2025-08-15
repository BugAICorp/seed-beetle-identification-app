""" simulator.py """
import sys
import os
from PIL import Image
from beetle_cropper import BeetleCropper
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram
from model_loader import ModelLoader
from evaluation_method import EvaluationMethod
from genus_evaluation_method import GenusEvaluationMethod
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
        print("Dorsal: 1\nCaudal: 2\nFrontal: 3\nLateral: 4")
        input = int(input("Choose a model you would like to train (type corresponding number): "))
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
        print("\nWould you like to train with an \"other\" class?")
        user_input = int(input("Enter 1 for YES, and 2 for NO: "))
        # if yes, set model paths to the "other" paths
        if user_input == 1:
            # set create other flag to true
            create_other = True

            # Species paths
            spec_caud_model = globals.spec_caud_model_with_other
            spec_dors_model = globals.spec_dors_model_with_other
            spec_fron_model = globals.spec_fron_model_with_other
            spec_late_model = globals.spec_late_model_with_other
            spec_class_dictionary = globals.spec_class_dictionary_with_other
            spec_accuracy_list = globals.spec_accuracy_list_with_other

            # Genus paths
            gen_caud_model = globals.gen_caud_model_with_other
            gen_dors_model = globals.gen_dors_model_with_other
            gen_fron_model = globals.gen_fron_model_with_other
            gen_late_model = globals.gen_late_model_with_other
            gen_class_dictionary = globals.gen_class_dictionary_with_other
            gen_accuracy_list = globals.gen_accuracy_list_with_other
            break
        # if no, set model paths to the normal paths
        if user_input == 2:
            # set create other flag to false
            create_other = False

            # Species paths
            spec_caud_model = globals.spec_caud_model
            spec_dors_model = globals.spec_dors_model
            spec_fron_model = globals.spec_fron_model
            spec_late_model = globals.spec_late_model
            spec_class_dictionary = globals.spec_class_dictionary
            spec_accuracy_list = globals.spec_accuracy_list

            # Genus paths
            gen_caud_model = globals.gen_caud_model
            gen_dors_model = globals.gen_dors_model
            gen_fron_model = globals.gen_fron_model
            gen_late_model = globals.gen_late_model
            gen_class_dictionary = globals.gen_class_dictionary
            gen_accuracy_list = globals.gen_accuracy_list
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
        print("\nWould you like to overwrite previous models no matter the new accuracy?")
        user_input = int(input("Enter 1 for YES, and 2 for NO: "))
        if user_input == 1:
            overwrite = True
            break
        if user_input == 2:
            overwrite = False
            break
        print("Invalid input. Please try again.")

    while True:
        # Ask user if they want to create per-class F1 score bar plot and a confusion matrix
        choice = input("\nWould you like to create Model performance visualizations? (y/n): ").lower()
        if choice == 'y':
            show_plots = True
            break
        if choice == 'n':
            show_plots = False
            break
        print("Invalid input. Please try again.")

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


    # initialize number of outputs
    SPECIES_OUTPUTS = dbr.get_num_species()
    GENUS_OUTPUTS = dbr.get_num_genus()

    # Run training with dataframe
    species_tp = TrainingProgram(df, "Species", SPECIES_OUTPUTS, augment = augment)

    # Training
    if train_caud:
        erasure_params_caud = {
            "p": 0.3978357251429255,
            "min": 0.04237603082954706,
            "max": 0.3025963685284483
        }
        species_tp.train_resnet_model(
            20, "caud", batch=16, rotation=1, brightness=0.0052372665532581155, lrate=0.000208665948737891,
            erasure_params=erasure_params_caud)
        if show_plots:
            species_tp.create_f1_scores_bar_plot(
                "caud", batch_size=32, plot_save_path="species_caud_plot.png", plot=True)
            species_tp.create_confusion_matrix(
                "caud", batch_size=32, plot_save_path="species_caud_matrix.png", plot=True)
    if train_dors:
        erasure_params_dors = {
            "p": 0.5763301129483613,
            "min": 0.06044662804540117,
            "max": 0.18387577071515754
        }
        species_tp.train_resnet_model(
            20, "dors", batch=16, rotation=8, brightness=0.11266599539746057, lrate=0.00016310975593889832,
            erasure_params=erasure_params_dors)
        if show_plots:
            species_tp.create_f1_scores_bar_plot(
                "dors", batch_size=32, plot_save_path="species_dors_plot.png", plot=True)
            species_tp.create_confusion_matrix(
                "dors", batch_size=32, plot_save_path="species_dors_matrix.png", plot=True)
    if train_fron:
        erasure_params_fron = {
            "p": 0.265585095702728,
            "min": 0.071779115882381,
            "max": 0.29234187228616554
        }
        species_tp.train_resnet_model(
            20, "fron", batch=32, rotation=12, brightness=0.14763773752336606, lrate=0.00018738820725043863,
            erasure_params=erasure_params_fron)
        if show_plots:
            species_tp.create_f1_scores_bar_plot(
                "fron", batch_size=32, plot_save_path="species_fron_plot.png", plot=True)
            species_tp.create_confusion_matrix(
                "fron", batch_size=32, plot_save_path="species_fron_matrix.png", plot=True)
    if train_late:
        erasure_params_late = {
            "p": 0.5189325280363017,
            "min": 0.03843699036307908,
            "max": 0.11129682877722781
        }
        species_tp.train_resnet_model(
            20, "late", batch=16, rotation=18, brightness=0.10813954357888121, lrate=0.0001659616690805536,
            erasure_params=erasure_params_late)
        if show_plots:
            species_tp.create_f1_scores_bar_plot(
                "late", batch_size=32, plot_save_path="species_late_plot.png", plot=True)
            species_tp.create_confusion_matrix(
                "late", batch_size=32, plot_save_path="species_late_matrix.png", plot=True)

    # Save models
    species_model_filenames = {
            "caud" : spec_caud_model if train_caud else None, 
            "dors" : spec_dors_model if train_dors else None,
            "fron" : spec_fron_model if train_fron else None,
            "late" : spec_late_model if train_late else None
        }

    species_tp.save_models(
        species_model_filenames,
        globals.img_height,
        spec_class_dictionary,
        spec_accuracy_list,
        overwrite)

    # Run training with dataframe
    genus_tp = TrainingProgram(df, "Genus", GENUS_OUTPUTS, augment = augment)

    # Training
    if train_caud:
        erasure_params_caud = {
            "p": 0.3127187908868738,
            "min": 0.04046194894255532,
            "max": 0.29175754421281885
        }
        genus_tp.train_resnet_model(
            20, "caud", batch=16, rotation=0, brightness=0.2983619077722387, lrate=0.00039800494669978446,
            erasure_params=erasure_params_caud)
        if show_plots:
            genus_tp.create_f1_scores_bar_plot(
                "caud", batch_size=32, plot_save_path="genus_caud_plot.png", plot=True)
            genus_tp.create_confusion_matrix(
                "caud", batch_size=32, plot_save_path="genus_caud_matrix.png", plot=True)

    if train_dors:
        erasure_params_dors = {
            "p": 0.08429225010786912,
            "min": 0.05881609232667761,
            "max": 0.29034641815208423
        }
        genus_tp.train_resnet_model(
            20, "dors", batch=16, rotation=9, brightness=0.05452949803911396, lrate=0.00034069432228042864,
            erasure_params=erasure_params_dors)
        if show_plots:
            genus_tp.create_f1_scores_bar_plot(
                "dors", batch_size=16, plot_save_path="genus_dors_plot.png", plot=True)
            genus_tp.create_confusion_matrix(
                "dors", batch_size=16, plot_save_path="genus_dors_matrix.png", plot=True)

    if train_fron:
        erasure_params_fron = {
            "p": 0.7558902433519469,
            "min": 0.07276752102604624,
            "max": 0.1953562902391759
        }
        genus_tp.train_resnet_model(
            20, "fron", batch=32, rotation=4, brightness=0.17667183838225514, lrate=0.0001997249630754838,
            erasure_params=erasure_params_fron)
        if show_plots:
            genus_tp.create_f1_scores_bar_plot(
                "fron", batch_size=16, plot_save_path="genus_fron_plot.png", plot=True)
            genus_tp.create_confusion_matrix(
                "fron", batch_size=16, plot_save_path="genus_fron_matrix.png", plot=True)

    if train_late:
        erasure_params_late = {
            "p": 0.3860968267885073,
            "min": 0.09392431854817945,
            "max": 0.2564630945836204
        }
        genus_tp.train_resnet_model(
            20, "late", batch=16, rotation=10, brightness=0.25458958614413363, lrate=0.00010421711239748923,
            erasure_params=erasure_params_late)
        if show_plots:
            genus_tp.create_f1_scores_bar_plot(
                "late", batch_size=32, plot_save_path="genus_late_plot.png", plot=True)
            genus_tp.create_confusion_matrix(
                "late", batch_size=32, plot_save_path="genus_late_matrix.png", plot=True)


    # Save models
    genus_model_filenames = {
        "caud" : gen_caud_model if train_caud else None, 
        "dors" : gen_dors_model if train_dors else None,
        "fron" : gen_fron_model if train_fron else None,
        "late" : gen_late_model if train_late else None
    }

    genus_tp.save_models(
        genus_model_filenames,
        globals.img_height,
        gen_class_dictionary,
        gen_accuracy_list,
        overwrite)

    # Load Genus models
    genus_model_paths = {
            "caud" : gen_caud_model, 
            "dors" : gen_dors_model,
            "fron" : gen_fron_model,
            "late" : gen_late_model
        }

    genus_ml = ModelLoader(genus_model_paths, GENUS_OUTPUTS)
    genus_models = genus_ml.get_models()

    print(genus_models.keys)
    print(genus_ml.get_model("caud").named_parameters())

    # Inititialize the EvaluationMethod object with the heaviest eval method set
    genus_evaluator = GenusEvaluationMethod(globals.img_height, genus_models, 1,
                                            gen_class_dictionary, gen_accuracy_list)

    # Get the images to be evaluated through user input
    LATE_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT LATE.jpg"
    DORS_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT DORS.jpg"
    FRON_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT FRON.jpg"
    CAUD_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT CAUD.jpg"

    # Load and crop the provided images
    LATE_IMG = beetle_cropper.crop_beetle(Image.open(LATE_PATH)) if LATE_PATH else None
    DORS_IMG = beetle_cropper.crop_beetle(Image.open(DORS_PATH)) if DORS_PATH else None
    FRON_IMG = beetle_cropper.crop_beetle(Image.open(FRON_PATH)) if FRON_PATH else None
    CAUD_IMG = beetle_cropper.crop_beetle(Image.open(CAUD_PATH)) if CAUD_PATH else None

    # Run the evaluation method to find the predicted genus
    top_genus, genus_conf_score = genus_evaluator.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Print classification results for genus
    print(f"Predicted Genus: {top_genus}, Confidence: {genus_conf_score:.2f}\n")

    # Load species models
    species_model_paths = {
            "caud" : spec_caud_model, 
            "dors" : spec_dors_model,
            "fron" : spec_fron_model,
            "late" : spec_late_model
        }
    species_ml = ModelLoader(species_model_paths, SPECIES_OUTPUTS)
    species_models = species_ml.get_models()

    print(species_models.keys)
    print(species_ml.get_model("caud").named_parameters())

    # Inititialize the EvaluationMethod object with the heaviest eval method set
    species_evaluator = EvaluationMethod(globals.img_height, species_models, 1,
                                         spec_class_dictionary, spec_accuracy_list)

    # Run the evaluation method
    top_5_species = species_evaluator.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Print classification results
    print(f"1. Predicted Species: {top_5_species[0][0]}, Confidence: {top_5_species[0][1]:.2f}\n")
    print(f"2. Predicted Species: {top_5_species[1][0]}, Confidence: {top_5_species[1][1]:.2f}\n")
    print(f"3. Predicted Species: {top_5_species[2][0]}, Confidence: {top_5_species[2][1]:.2f}\n")
    print(f"4. Predicted Species: {top_5_species[3][0]}, Confidence: {top_5_species[3][1]:.2f}\n")
    print(f"5. Predicted Species: {top_5_species[4][0]}, Confidence: {top_5_species[4][1]:.2f}\n")
