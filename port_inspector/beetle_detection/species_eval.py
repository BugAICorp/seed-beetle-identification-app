# flake8: noqa
import json, os, sys


# -----Run once on server start up-----
if "runserver" in sys.argv:
    from PIL import Image
    from .model_loader import ModelLoader
    from .evaluation_method import EvaluationMethod
    from .genus_evaluation_method import GenusEvaluationMethod
    from .data_converter import DjangoTrainingDatabaseConverter
    from .app_training_program import TrainingProgram
    from django.conf import settings

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    import globals

    # read json to see size of outputs
    spec_dict_path = os.path.join(BASE_DIR, "spec_dict.json")
    with open(spec_dict_path, 'r', encoding='utf-8') as spec_dict:
        SPECIES_OUTPUTS = len(json.load(spec_dict))

    gen_dict_path = os.path.join(BASE_DIR, "gen_dict.json")
    with open(gen_dict_path, 'r', encoding='utf-8') as gen_dict:
        GENUS_OUTPUTS = len(json.load(gen_dict))

    # Load species models
    species_model_paths = {
            "caud" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_caud.pth"), 
            "dors" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_dors.pth"),
            "fron" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_fron.pth"),
            "late" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_late.pth")
        }
    species_ml = ModelLoader(species_model_paths, SPECIES_OUTPUTS)
    species_models = species_ml.get_models()

    # Load genus models
    genus_model_paths = {
            "caud" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_caud.pth"), 
            "dors" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_dors.pth"),
            "fron" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_fron.pth"),
            "late" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_late.pth")
        }
    genus_ml = ModelLoader(genus_model_paths, GENUS_OUTPUTS)
    genus_models = genus_ml.get_models()


    # Initialize the EvaluationMethod object with the heaviest eval method set
    species_evaluator = EvaluationMethod(os.path.join(os.path.dirname(os.path.abspath(__file__)), "height.txt"), species_models, 1, 
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_dict.json"), 
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_accuracies.json"))
    genus_evaluator = GenusEvaluationMethod(os.path.join(os.path.dirname(os.path.abspath(__file__)), "height.txt"), genus_models, 1, 
                                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_dict.json"), 
                                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_accuracies.json"))

    print("!!! ML Models loaded in evaluation mode !!!")

    dbr = DjangoTrainingDatabaseConverter("dataset")
    dbr.conversion()


def evaluate_images(late_path, dors_path, fron_path, caud_path):
    # Load the provided images
    LATE_IMG = Image.open(late_path) if late_path else None
    DORS_IMG = Image.open(dors_path) if dors_path else None
    FRON_IMG = Image.open(fron_path) if fron_path else None
    CAUD_IMG = Image.open(caud_path) if caud_path else None    
    
    # Run the species evaluation method
    top_species = species_evaluator.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Run the genus evaluation method
    top_genus = genus_evaluator.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Print classification results
    print(f"1. Predicted Species: {top_species[0][0]}, Confidence: {top_species[0][1]:.5f}\n")
    print(f"2. Predicted Species: {top_species[1][0]}, Confidence: {top_species[1][1]:.5f}\n")
    print(f"3. Predicted Species: {top_species[2][0]}, Confidence: {top_species[2][1]:.5f}\n")
    print(f"4. Predicted Species: {top_species[3][0]}, Confidence: {top_species[3][1]:.5f}\n")
    print(f"5. Predicted Species: {top_species[4][0]}, Confidence: {top_species[4][1]:.5f}\n")
    
    print("Top genus: ", top_genus)

    # Take top 5 species, modify confidence numbers to be a percentage, Ensure name strings are in title format
    top_5_species = []
    for i in range(5):
        top_5_species.append((top_species[i][0].title(), top_species[i][1]*100.0))
    top_genus = top_genus[0], top_genus[1]*100.0

    return top_5_species, top_genus


def retrain_models():
    """
    Script for running retraining of models
    """
    species_tp = TrainingProgram('species', 'image')
    genus_tp = TrainingProgram('genus', 'image')

    species_tp.train_resnet_model(20, 'late')
    genus_tp.train_resnet_model(20, 'late')
    species_tp.train_resnet_model(20, 'dors')
    genus_tp.train_resnet_model(20, 'dors')
    species_tp.train_resnet_model(20, 'fron')
    genus_tp.train_resnet_model(20, 'fron')
    species_tp.train_resnet_model(20, 'caud')
    genus_tp.train_resnet_model(20, 'caud')

    species_model_filenames = {
        "caud" : globals.spec_caud_model,
        "dors" : globals.spec_dors_model,
        "fron" : globals.spec_fron_model,
        "late" : globals.spec_late_model
    }

    species_tp.save_models(
        species_model_filenames,
        globals.img_height,
        globals.spec_class_dictionary,
        globals.spec_accuracy_list
    )


def refresh_database():
    """
    Method to be called upon saving a new valid class
    """
    dbr.conversion()
