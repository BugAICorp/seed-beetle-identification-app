""" simulator.py """
import sys
import os
from PIL import Image
from beetle_cropper import BeetleCropper
from training_database_creator import TrainingDataConverter
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
        print("Dorsal: 1\nCaudal: 2\nFrontal: 3\nLateral: 4\nAll: 5")
        user_input = int(input("Choose a model you would like to train (type corresponding number): "))
        if user_input == 1:
            train_dors = True
        elif user_input == 2:
            train_caud = True
        elif user_input == 3:
            train_fron = True
        elif user_input == 4:
            train_late = True
        elif user_input == 5:
            train_dors = True
            train_caud = True
            train_fron = True
            train_late = True
        else:
            print("Invalid Input")
        del user_input
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
        # Ask user if they want to run uncertainty evaluation after training has completed
        choice = input("\nWould you like to evaluate the uncertainty of the model(s) after training? (y/n): ").lower()
        if choice == 'y':
            uncertainty_eval = True
            break
        if choice == 'n':
            uncertainty_eval = False
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

    if show_plots:
        while True:
            # Ask user which type of confusion matrix they would like to generate
            print("\nWhich confusion matrix would you like to generate?")
            print("\t0 = Row-normalized Recall Values\n" \
                "\t1 = Raw Counts\n" \
                "\t2 = Both (Normalized Recall and Raw Counts)")
            choice = int(input("Enter the number of your choice: "))
            if choice == 0:
                recall = True
                break
            if choice == 1:
                raw_counts = True
                break
            if choice == 2:
                recall = True
                raw_counts = True
                break
            print("Invalid Input. Please enter 0, 1, or 2.")
    while True:
        # Ask user which evaluation method they would like to use
        choice = input("\nWould you like to evaluate using Monte Carlo Dropout? (y/n): ").lower()
        if choice == 'y':
            mc_eval = True
            break
        if choice == 'n':
            mc_eval = False
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
    species_tp = TrainingProgram(df, "Species", SPECIES_OUTPUTS, augment = augment, balance_classes=balance_classes)

    # Training
    if train_caud:
        erasure_params_caud = {
            "p": 0.5450068594306283,
            "min": 0.032231275920186486,
            "max": 0.23975077356392424
        }
        species_tp.train_resnet_model(20, "caud", batch=16, rotation=6,
                                    brightness=0.0672682540489113, weight_decay=0.1, lrate=0.0002205207835665262,
                                    erasure_params=erasure_params_caud, max_os_ratio=1.5)

        if uncertainty_eval:
            print("\nRunning Monte Carlo Dropout uncertainty evaluation for species CAUD view...")
            results = species_tp.evaluate_uncertainty(
                view="caud",
                n_samples=20,
                batch_size=16,
                threshold=0.02
            )

            # Print quick summary
            avg_uncertainty = sum(results["all_uncertainties"]) / len(results["all_uncertainties"])
            print(f"Average uncertainty across test set: {avg_uncertainty:.4f}")
            kept = len(results["filtered_preds"])
            total = len(results["all_preds"])
            print(f"Total predictions kept after thresholding: {kept}/{total}")

        if show_plots:
            species_tp.create_f1_scores_bar_plot(
                "caud", batch_size=16, plot_save_path="species_caud_plot.png", plot=True)
            if recall:
                species_tp.create_confusion_matrix(
                    "caud", batch_size=16, plot_save_path="species_caud_matrix_recall.png", plot=True, normalize=True)
            if raw_counts:
                species_tp.create_confusion_matrix(
                    "caud", batch_size=16, plot_save_path="species_caud_matrix_counts.png", plot=True, normalize=False)

    if train_dors:
        erasure_params_dors = {
            "p": 0.7757711509313643,
            "min": 0.01008374654178916,
            "max": 0.38794012670750844
        }
        species_tp.train_resnet_model(20, "dors", batch=16, rotation=12,
                                    brightness=0.22216817398095146, weight_decay=0.1, lrate=0.0001296278789334687,
                                    erasure_params=erasure_params_dors, max_os_ratio=2.5)

        if uncertainty_eval:
            print("\nRunning Monte Carlo Dropout uncertainty evaluation for species DORS view...")
            results = species_tp.evaluate_uncertainty(
                view="dors",
                n_samples=20,
                batch_size=16,
                threshold=0.02
            )

            avg_uncertainty = sum(results["all_uncertainties"]) / len(results["all_uncertainties"])
            print(f"Average uncertainty across test set: {avg_uncertainty:.4f}")
            kept = len(results["filtered_preds"])
            total = len(results["all_preds"])
            print(f"Total predictions kept after thresholding: {kept}/{total}")

        if show_plots:
            species_tp.create_f1_scores_bar_plot(
                "dors", batch_size=16, plot_save_path="species_dors_plot.png", plot=True)
            if recall:
                species_tp.create_confusion_matrix(
                    "dors", batch_size=16, plot_save_path="species_dors_matrix_recall.png", plot=True, normalize=True)
            if raw_counts:
                species_tp.create_confusion_matrix(
                    "dors", batch_size=16, plot_save_path="species_dors_matrix_counts.png", plot=True, normalize=False)

    if train_fron:
        erasure_params_fron = {
            "p": 0.14786083200104405,
            "min": 0.08542272176573411,
            "max": 0.3766890143419105
        }
        species_tp.train_resnet_model(20, "fron", batch=16, rotation=7,
                                    brightness=0.16052298566019538, weight_decay=0.1, lrate=0.00018151090290770348,
                                    erasure_params=erasure_params_fron, max_os_ratio=4.0)

        if uncertainty_eval:
            print("\nRunning Monte Carlo Dropout uncertainty evaluation for species FRON view...")
            results = species_tp.evaluate_uncertainty(
                view="fron",
                n_samples=20,
                batch_size=16,
                threshold=0.02
            )

            avg_uncertainty = sum(results["all_uncertainties"]) / len(results["all_uncertainties"])
            print(f"Average uncertainty across test set: {avg_uncertainty:.4f}")
            kept = len(results["filtered_preds"])
            total = len(results["all_preds"])
            print(f"Total predictions kept after thresholding: {kept}/{total}")

        if show_plots:
            species_tp.create_f1_scores_bar_plot(
                "fron", batch_size=16, plot_save_path="species_fron_plot.png", plot=True)
            if recall:
                species_tp.create_confusion_matrix(
                    "fron", batch_size=16, plot_save_path="species_fron_matrix_recall.png", plot=True, normalize=True)
            if raw_counts:
                species_tp.create_confusion_matrix(
                    "fron", batch_size=16, plot_save_path="species_fron_matrix_counts.png", plot=True, normalize=False)

    if train_late:
        erasure_params_late = {
            "p": 0.005799105801707227,
            "min": 0.08818090418966613,
            "max": 0.2566152645216
        }
        species_tp.train_resnet_model(20, "late", batch=64, rotation=6,
                                    brightness=0.29977566775503983, weight_decay=0.1, lrate=0.00012089084719947084,
                                    erasure_params=erasure_params_late, max_os_ratio=3.5)

        if uncertainty_eval:
            print("\nRunning Monte Carlo Dropout uncertainty evaluation for species LATE view...")
            results = species_tp.evaluate_uncertainty(
                view="late",
                n_samples=20,
                batch_size=64,
                threshold=0.02
            )

            avg_uncertainty = sum(results["all_uncertainties"]) / len(results["all_uncertainties"])
            print(f"Average uncertainty across test set: {avg_uncertainty:.4f}")
            kept = len(results["filtered_preds"])
            total = len(results["all_preds"])
            print(f"Total predictions kept after thresholding: {kept}/{total}")

        if show_plots:
            species_tp.create_f1_scores_bar_plot(
                "late", batch_size=64, plot_save_path="species_late_plot.png", plot=True)
            if recall:
                species_tp.create_confusion_matrix(
                    "late", batch_size=64, plot_save_path="species_late_matrix_recall.png", plot=True, normalize=True)
            if raw_counts:
                species_tp.create_confusion_matrix(
                    "late", batch_size=64, plot_save_path="species_late_matrix_counts.png", plot=True, normalize=False)

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
    genus_tp = TrainingProgram(df, "Genus", GENUS_OUTPUTS, augment = augment, balance_classes=balance_classes)

    # Training
    if train_caud:
        erasure_params_caud = {
            "p": 0.117534992000064,
            "min": 0.08054270560117567,
            "max": 0.2983577819330524
        }
        genus_tp.train_resnet_model(20, "caud", batch=16, rotation=10,
                                brightness=0.1462847736327197, weight_decay=0.1, lrate=0.00004409398823911199,
                                erasure_params=erasure_params_caud, max_os_ratio=5.0)

        if uncertainty_eval:
            print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus CAUD view...")
            results = genus_tp.evaluate_uncertainty(
                view="caud",
                n_samples=20,
                batch_size=16,
                threshold=0.02
            )

            avg_uncertainty = sum(results["all_uncertainties"]) / len(results["all_uncertainties"])
            print(f"Average uncertainty across test set: {avg_uncertainty:.4f}")
            kept = len(results["filtered_preds"])
            total = len(results["all_preds"])
            print(f"Total predictions kept after thresholding: {kept}/{total}")

        if show_plots:
            genus_tp.create_f1_scores_bar_plot(
                "caud", batch_size=16, plot_save_path="genus_caud_plot.png", plot=True)
            if recall:
                genus_tp.create_confusion_matrix(
                    "caud", batch_size=16, plot_save_path="genus_caud_matrix_recall.png", plot=True, normalize=True)
            if raw_counts:
                genus_tp.create_confusion_matrix(
                    "caud", batch_size=16, plot_save_path="genus_caud_matrix_counts.png", plot=True, normalize=False)

    if train_dors:
        erasure_params_dors = {
            "p": 0.6279748323341047,
            "min": 0.041921505805665914,
            "max": 0.24388226488220693
        }
        genus_tp.train_resnet_model(20, "dors", batch=32, rotation=6,
                                brightness=0.2988104061389692, weight_decay=0.1, lrate=0.00004736821824349854,
                                erasure_params=erasure_params_dors, max_os_ratio=1.0)

        if uncertainty_eval:
            print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus DORS view...")
            results = genus_tp.evaluate_uncertainty(
                view="dors",
                n_samples=20,
                batch_size=32,
                threshold=0.02
            )

            avg_uncertainty = sum(results["all_uncertainties"]) / len(results["all_uncertainties"])
            print(f"Average uncertainty across test set: {avg_uncertainty:.4f}")
            kept = len(results["filtered_preds"])
            total = len(results["all_preds"])
            print(f"Total predictions kept after thresholding: {kept}/{total}")

        if show_plots:
            genus_tp.create_f1_scores_bar_plot(
                "dors", batch_size=32, plot_save_path="genus_dors_plot.png", plot=True)
            if recall:
                genus_tp.create_confusion_matrix(
                    "dors", batch_size=32, plot_save_path="genus_dors_matrix_recall.png", plot=True, normalize=True)
            if raw_counts:
                genus_tp.create_confusion_matrix(
                    "dors", batch_size=32, plot_save_path="genus_dors_matrix_counts.png", plot=True, normalize=False)

    if train_fron:
        erasure_params_fron = {
            "p": 0.30518586009082976,
            "min": 0.04609315007975057,
            "max": 0.36140797065499464
        }
        genus_tp.train_resnet_model(20, "fron", batch=64, rotation=14,
                                brightness=0.22903306674663448, weight_decay=0.1, lrate=0.0001380146193447115,
                                erasure_params=erasure_params_fron, max_os_ratio=5.0)

        if uncertainty_eval:
            print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus FRON view...")
            results = genus_tp.evaluate_uncertainty(
                view="fron",
                n_samples=20,
                batch_size=64,
                threshold=0.02
            )

            avg_uncertainty = sum(results["all_uncertainties"]) / len(results["all_uncertainties"])
            print(f"Average uncertainty across test set: {avg_uncertainty:.4f}")
            kept = len(results["filtered_preds"])
            total = len(results["all_preds"])
            print(f"Total predictions kept after thresholding: {kept}/{total}")

        if show_plots:
            genus_tp.create_f1_scores_bar_plot(
                "fron", batch_size=64, plot_save_path="genus_fron_plot.png", plot=True)
            if recall:
                genus_tp.create_confusion_matrix(
                    "fron", batch_size=64, plot_save_path="genus_fron_matrix_recall.png", plot=True, normalize=True)
            if raw_counts:
                genus_tp.create_confusion_matrix(
                    "fron", batch_size=64, plot_save_path="genus_fron_matrix_counts.png", plot=True, normalize=False)

    if train_late:
        erasure_params_late = {
            "p": 0.30535724516213314,
            "min": 0.011359991265195598,
            "max": 0.31162030351760406
        }
        genus_tp.train_resnet_model(20, "late", batch=32, rotation=10,
                                brightness=0.04304050259182124, weight_decay=0.1, lrate=0.00001826137626671228,
                                erasure_params=erasure_params_late, max_os_ratio=3.0)

        if uncertainty_eval:
            print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus LATE view...")
            results = genus_tp.evaluate_uncertainty(
                view="late",
                n_samples=20,
                batch_size=32,
                threshold=0.02
            )

            avg_uncertainty = sum(results["all_uncertainties"]) / len(results["all_uncertainties"])
            print(f"Average uncertainty across test set: {avg_uncertainty:.4f}")
            kept = len(results["filtered_preds"])
            total = len(results["all_preds"])
            print(f"Total predictions kept after thresholding: {kept}/{total}")

        if show_plots:
            genus_tp.create_f1_scores_bar_plot(
                "late", batch_size=32, plot_save_path="genus_late_plot.png", plot=True)
            if recall:
                genus_tp.create_confusion_matrix(
                    "late", batch_size=32, plot_save_path="genus_late_matrix_recall.png", plot=True, normalize=True)
            if raw_counts:
                genus_tp.create_confusion_matrix(
                    "late", batch_size=32, plot_save_path="genus_late_matrix_counts.png", plot=True, normalize=False)


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

    genus_ml = ModelLoader(genus_model_paths, GENUS_OUTPUTS, use_dropout=True)
    genus_models = genus_ml.get_models()

    print(genus_models.keys())
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
    if mc_eval:
        mc_genus_eval = genus_evaluator.evaluate_image_mc_dropout(
            late=LATE_IMG,
            dors=DORS_IMG,
            fron=FRON_IMG,
            caud=CAUD_IMG,
            n_samples=20)

        for view in [ "caud", "dors", "fron", "late"]:
            if mc_genus_eval[view]:
                genus_eval_dict = mc_genus_eval[view]
                top_genus = genus_eval_dict["genus"]
                genus_conf_score = genus_eval_dict["score"]
                genus_uncertainty = genus_eval_dict["uncertainty"]
                # Print classification results for genus
                print(f"Monte Carlo Dropout Genus Evaluation Results for {view} View:")
                print(f"\tPredicted Genus: {top_genus}, Confidence: {genus_conf_score:.2f}\n")
                print(f"\tUncertainty for {view} View: {genus_uncertainty:.4f}\n")
    else:
        top_genus, genus_conf_score = genus_evaluator.evaluate_image(
            late=LATE_IMG,
            dors=DORS_IMG,
            fron=FRON_IMG,
            caud=CAUD_IMG)

        # Print classification results for genus
        print(f"Predicted Genus: {top_genus}, Confidence: {genus_conf_score:.2f}\n")

    # Load species models
    species_model_paths = {
            "caud" : spec_caud_model, 
            "dors" : spec_dors_model,
            "fron" : spec_fron_model,
            "late" : spec_late_model
        }
    species_ml = ModelLoader(species_model_paths, SPECIES_OUTPUTS, use_dropout=True)
    species_models = species_ml.get_models()

    print(species_models.keys())
    print(species_ml.get_model("caud").named_parameters())

    # Inititialize the EvaluationMethod object with the heaviest eval method set
    species_evaluator = EvaluationMethod(globals.img_height, species_models, 1,
                                         spec_class_dictionary, spec_accuracy_list)

    # Run the evaluation method
    if mc_eval:
        mc_species_eval = species_evaluator.evaluate_image_mc_dropout(
            late=LATE_IMG,
            dors=DORS_IMG,
            fron=FRON_IMG,
            caud=CAUD_IMG,
            n_samples=20)

        for view in [ "caud", "dors", "fron", "late"]:
            if mc_species_eval[view]:
                species_eval_dict = mc_species_eval[view]
                top_5_species = species_eval_dict["species"]
                species_conf_scores = species_eval_dict["mean_scores"]
                species_uncertainty = species_eval_dict["uncertainty"]

                print(f"Monte Carlo Dropout Species Evaluation Results for {view} View:")
                for i, species_name in enumerate(top_5_species):
                    species_conf = species_conf_scores[i]
                    print(f"\t{i+1}. Predicted Species: {species_name}, Confidence: {species_conf:.2f}")
                print(f"Uncertainty for {view} view: {species_uncertainty:.4f}\n")

    else:
        top_5_species = species_evaluator.evaluate_image(
            late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
        )
        # Print classification results
        print(f"1. Predicted Species: {top_5_species[0][0]}, Confidence: {top_5_species[0][1]:.2f}\n")
        print(f"2. Predicted Species: {top_5_species[1][0]}, Confidence: {top_5_species[1][1]:.2f}\n")
        print(f"3. Predicted Species: {top_5_species[2][0]}, Confidence: {top_5_species[2][1]:.2f}\n")
        print(f"4. Predicted Species: {top_5_species[3][0]}, Confidence: {top_5_species[3][1]:.2f}\n")
        print(f"5. Predicted Species: {top_5_species[4][0]}, Confidence: {top_5_species[4][1]:.2f}\n")
