# flake8: noqa
import json, os, sys


# -----Run once on server start up-----
import io
import redis
import torch
import gc
from PIL import Image
from .model_loader import ModelLoader
from .evaluation_method import EvaluationMethod
from .genus_evaluation_method import GenusEvaluationMethod
from .data_converter import DjangoTrainingDatabaseConverter
from .app_training_program import TrainingProgram
from django.conf import settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import globals

dbr = DjangoTrainingDatabaseConverter("dataset")

# Global redis connection
redis_connection = None

# --- Lazy global references ---
species_evaluator = None
genus_evaluator = None
_models_loaded = False

def get_redis_conn():
    global redis_connection
    if redis_connection is None:
        redis_url = "redis://localhost:6379/0"
        redis_connection = redis.from_url(
            redis_url,
            decode_responses=False
        )
    return redis_connection

def load_models_once():
    global species_evaluator, genus_evaluator, _models_loaded
    if _models_loaded:
        return species_evaluator, genus_evaluator
    
    # limit threading inside process
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


    # read json to see size of outputs
    spec_dict_path = os.path.join(BASE_DIR, "model_data/spec_dict.json")
    with open(spec_dict_path, 'r', encoding='utf-8') as spec_dict:
        SPECIES_OUTPUTS = len(json.load(spec_dict))

    gen_dict_path = os.path.join(BASE_DIR, "model_data/gen_dict.json")
    with open(gen_dict_path, 'r', encoding='utf-8') as gen_dict:
        GENUS_OUTPUTS = len(json.load(gen_dict))

    # Load species models
    species_model_paths = {
        "caud" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/spec_caud.pth"), 
        "dors" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/spec_dors.pth"),
        "fron" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/spec_fron.pth"),
        "late" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/spec_late.pth")
    }

    # Load genus models
    genus_model_paths = {
        "caud" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/gen_caud.pth"), 
        "dors" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/gen_dors.pth"),
        "fron" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/gen_fron.pth"),
        "late" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/gen_late.pth")
    }

    species_ml = ModelLoader(
        weights_file_paths=species_model_paths, num_classes=SPECIES_OUTPUTS, use_dropout=True)

    genus_ml = ModelLoader(
        weights_file_paths=genus_model_paths, num_classes=GENUS_OUTPUTS, use_dropout=True)

    species_models = species_ml.get_models()
    genus_models = genus_ml.get_models()


    # Initialize the EvaluationMethod object with the heaviest eval method set
    species_evaluator = EvaluationMethod(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data/height.txt"), species_models, 1, 
                                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data/spec_dict.json"), 
                                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data/spec_accuracies.json"))
    genus_evaluator = GenusEvaluationMethod(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data/height.txt"), genus_models, 1, 
                                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data/gen_dict.json"), 
                                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data/gen_accuracies.json"))

    print("!!! ML Models loaded in evaluation mode !!!")
    _models_loaded = True
    return species_evaluator, genus_evaluator

def evaluate_images(upload_id):
    species_eval, genus_eval = load_models_once()

    redis_conn = get_redis_conn()
    if not redis_conn:
        return [], 0  # Redis not configured, skip

    # Fetch images from Redis
    view_images = {}
    for view in ["lateral", "dorsal", "frontal", "caudal"]:
        img_bytes = redis_conn.get(f"upload:{upload_id}:{view}")
        if img_bytes:
            view_images[view] = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            view_images[view] = None

    LATE_IMG = view_images["lateral"]
    DORS_IMG = view_images["dorsal"]
    FRON_IMG = view_images["frontal"]
    CAUD_IMG = view_images["caudal"]

    try:
        # Run the species evaluation method
        top_species = species_eval.evaluate_image(
            late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
        )

        # Run the genus evaluation method
        top_genus = genus_eval.evaluate_image(
            late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
        )
    finally:
        # make sure to close PIL images
        for im in (LATE_IMG, DORS_IMG, FRON_IMG, CAUD_IMG):
            if hasattr(im, "close"):
                try:
                    im.close()
                except Exception:
                    pass

        # force gc and empty cache
        del LATE_IMG, DORS_IMG, FRON_IMG, CAUD_IMG
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        top_5_species.append((top_species[i][0], top_species[i][1]*100.0))
    top_genus = top_genus[0], top_genus[1]*100.0

    return top_5_species, top_genus


def evaluate_mc_dropout(upload_id):
    """
    Script to evaluate via mc dropout the input images and process the results
    for the app to display
    """
    species_eval, genus_eval = load_models_once()

    redis_conn = get_redis_conn()
    if not redis_conn:
        return [], 0  # Redis not configured, skip

    # Fetch images from Redis
    view_images = {}
    for view in ["lateral", "dorsal", "frontal", "caudal"]:
        img_bytes = redis_conn.get(f"upload:{upload_id}:{view}")
        if img_bytes:
            view_images[view] = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            view_images[view] = None

    LATE_IMG = view_images["lateral"]
    DORS_IMG = view_images["dorsal"]
    FRON_IMG = view_images["frontal"]
    CAUD_IMG = view_images["caudal"]

    try:
        # Run the species evaluation method
        top_species = species_eval.evaluate_heaviest_mc_dropout(
            late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
        )

        top_genus = genus_eval.evaluate_heaviest_mc_dropout(
            late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
        )
    finally:
        # make sure to close PIL images
        for im in (LATE_IMG, DORS_IMG, FRON_IMG, CAUD_IMG):
            if hasattr(im, "close"):
                try:
                    im.close()
                except Exception:
                    pass

        # force gc and empty cache
        del LATE_IMG, DORS_IMG, FRON_IMG, CAUD_IMG
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    top_genus_formatted = top_genus["genus"], top_genus["mean_score"]*100.0
    top_5_species = []
    for i in range(5):
        top_5_species.append((top_species["species"][i], top_species["mean_scores"][i]*100.0))

    return top_5_species, top_genus_formatted, top_species["uncertainty"], top_species["status"], top_genus["uncertainty"], top_genus["status"]


def retrain_models():
    """
    Script for running retraining of models
    """
    species_tp = TrainingProgram('species', 'image', augment=True, balance_classes=2)
    genus_tp = TrainingProgram('genus', 'image', augment=True, balance_classes=2)

    # Species LATE
    erasure_params_late = {
        "p": 0.005799105801707227,
        "min": 0.08818090418966613,
        "max": 0.2566152645216
    }
    species_tp.train_resnet_model(20, "late", batch=64, rotation=6,
                                  brightness=0.29977566775503983, lrate=0.00012089084719947084,
                                  erasure_params=erasure_params_late, max_os_ratio=3.5)

    # Genus LATE
    erasure_params_late = {
        "p": 0.30535724516213314,
        "min": 0.011359991265195598,
        "max": 0.31162030351760406
    }
    genus_tp.train_resnet_model(20, "late", batch=32, rotation=10,
                                brightness=0.04304050259182124, lrate=0.00001826137626671228,
                                erasure_params=erasure_params_late, max_os_ratio=3.0)

    # Species DORS
    erasure_params_dors = {
        "p": 0.7757711509313643,
        "min": 0.01008374654178916,
        "max": 0.38794012670750844
    }
    species_tp.train_resnet_model(20, "dors", batch=16, rotation=12,
                                  brightness=0.22216817398095146, lrate=0.0001296278789334687,
                                  erasure_params=erasure_params_dors, max_os_ratio=2.5)

    # Genus DORS
    erasure_params_dors = {
        "p": 0.6279748323341047,
        "min": 0.041921505805665914,
        "max": 0.24388226488220693
    }
    genus_tp.train_resnet_model(20, "dors", batch=32, rotation=6,
                                brightness=0.2988104061389692, lrate=0.00004736821824349854,
                                erasure_params=erasure_params_dors, max_os_ratio=1.0)

    # Species FRON
    erasure_params_fron = {
        "p": 0.14786083200104405,
        "min": 0.08542272176573411,
        "max": 0.3766890143419105
    }
    species_tp.train_resnet_model(20, "fron", batch=16, rotation=7,
                                  brightness=0.16052298566019538, lrate=0.00018151090290770348,
                                  erasure_params=erasure_params_fron, max_os_ratio=4.0)

    # Genus FRON
    erasure_params_fron = {
        "p": 0.30518586009082976,
        "min": 0.04609315007975057,
        "max": 0.36140797065499464
    }
    genus_tp.train_resnet_model(20, "fron", batch=64, rotation=14,
                                brightness=0.22903306674663448, lrate=0.0001380146193447115,
                                erasure_params=erasure_params_fron, max_os_ratio=5.0)

    # Species CAUD
    erasure_params_caud = {
        "p": 0.5450068594306283,
        "min": 0.032231275920186486,
        "max": 0.23975077356392424
    }
    species_tp.train_resnet_model(20, "caud", batch=16, rotation=6,
                                  brightness=0.0672682540489113, lrate=0.0002205207835665262,
                                  erasure_params=erasure_params_caud, max_os_ratio=1.5)

    # Genus CAUD
    erasure_params_caud = {
        "p": 0.117534992000064,
        "min": 0.08054270560117567,
        "max": 0.2983577819330524
    }
    genus_tp.train_resnet_model(20, "caud", batch=16, rotation=10,
                                brightness=0.1462847736327197, lrate=0.00004409398823911199,
                                erasure_params=erasure_params_caud, max_os_ratio=5.0)

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
