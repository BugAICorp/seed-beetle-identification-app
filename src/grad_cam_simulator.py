""" grad_cam_simulator.py """

import os
import sys
import json
from PIL import Image
import torch
import dill
from torchvision import transforms
from model_visualizer import GradCAMVisualizer
from model_loader import ModelLoader
from beetle_cropper import BeetleCropper
import globals

def get_image_path(view_name, default_path):
    """
    Prompt user for an image path, or use the default if none is entered.
    """
    path = input(f"Enter path to {view_name.upper()} image or press ENTER for [{default_path}]: ").strip()
    return "dataset/"+path if path else default_path

def load_and_crop_images(paths_dict):
    """
    Crop the provided image paths using the BeetleCropper.
    """
    cropper = BeetleCropper()
    cropped = {}
    for view, path in paths_dict.items():
        if not os.path.exists(path):
            print(f"Warning! Image for {view} not found at {path}. Skipping.")
            continue
        img = Image.open(path).convert("RGB")
        cropped[view] = cropper.crop_beetle(img)
    return cropped

def run_gradcam(models, mode_name, images_dict, transform_dict):
    """
    Run Grad-CAM and save visualizations.
    """
    for view, model in models.items():
        if view not in images_dict:
            print(f"Skipping {view} view for {mode_name} due to missing image.")
            continue

        print(f"Generating Grad-CAM for {mode_name} model: {view}")
        original_image = images_dict[view]
        transform = transform_dict[view]
        image_tensor = transform(original_image)

        last_conv_layer = model.layer4[-1]  # Assumes ResNet50

        visualizer = GradCAMVisualizer(model, target_layer=last_conv_layer)

        output_path = f"grad_cam_outputs/{view}_{mode_name}.png"
        visualizer.save_visualization(
            image_tensor=image_tensor,
            original_image=original_image,
            output_path=output_path,
            title=f"Grad-CAM: {view} ({mode_name.capitalize()})"
        )
        print(f"Saved {mode_name.capitalize()} Grad-CAM to {output_path}")


if __name__ == '__main__':
    # Ask the user which model(s) to run
    choice = input("Run Grad-CAM for (genus/species/both)? ").strip().lower()
    run_genus = choice in ("genus", "both")
    run_species = choice in ("species", "both")

    if not (run_genus or run_species):
        print("Invalid choice. Exiting.")
        sys.exit(1)

    while True:
        print("\nWould you like to use the models trained with an \"other\" class?")
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

            # Genus paths
            gen_caud_model = globals.gen_caud_model_with_other
            gen_dors_model = globals.gen_dors_model_with_other
            gen_fron_model = globals.gen_fron_model_with_other
            gen_late_model = globals.gen_late_model_with_other
            gen_class_dictionary = globals.gen_class_dictionary_with_other
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

            # Genus paths
            gen_caud_model = globals.gen_caud_model
            gen_dors_model = globals.gen_dors_model
            gen_fron_model = globals.gen_fron_model
            gen_late_model = globals.gen_late_model
            gen_class_dictionary = globals.gen_class_dictionary
            break
        print("Invalid Input. Please enter 1 or 2.")

    # Load dictionaries to count output classes
    with open(spec_class_dictionary, "r") as f:
        spec_dict = json.load(f)
    SPECIES_OUTPUTS = len(spec_dict)

    with open(gen_class_dictionary, "r") as f:
        gen_dict = json.load(f)
    GENUS_OUTPUTS = len(gen_dict)

    # Image paths for each view (default or user-provided)
    image_paths = {
        "late": get_image_path("late", "dataset/Callosobruchus chinensis GEM_187686348 5XEXT LATE.jpg"),
        "dors": get_image_path("dors", "dataset/Callosobruchus chinensis GEM_187686348 5XEXT DORS.jpg"),
        "fron": get_image_path("fron", "dataset/Callosobruchus chinensis GEM_187686348 5XEXT FRON.jpg"),
        "caud": get_image_path("caud", "dataset/Callosobruchus chinensis GEM_187686348 5XEXT CAUD.jpg"),
    }

    # Crop images
    cropped_images = load_and_crop_images(image_paths)

    # Load transformations
    transformations = {}
    with open(globals.caud_transformation, "rb") as f:
        transformations["caud"] = dill.load(f)
    with open(globals.dors_transformation, "rb") as f:
        transformations["dors"] = dill.load(f)
    with open(globals.fron_transformation, "rb") as f:
        transformations["fron"] = dill.load(f)
    with open(globals.late_transformation, "rb") as f:
        transformations["late"] = dill.load(f)

    # Load and run species models
    if run_species:
        species_model_filenames = {
            "caud": spec_caud_model,
            "dors": spec_dors_model,
            "fron": spec_fron_model,
            "late": spec_late_model
        }
        species_ml = ModelLoader(species_model_filenames, SPECIES_OUTPUTS)
        species_models = species_ml.get_models()
        run_gradcam(species_models, "species", cropped_images, transformations)

    # Load and run genus models
    if run_genus:
        genus_model_filenames = {
            "caud": gen_caud_model,
            "dors": gen_dors_model,
            "fron": gen_fron_model,
            "late": gen_late_model
        }
        genus_ml = ModelLoader(genus_model_filenames, GENUS_OUTPUTS)
        genus_models = genus_ml.get_models()
        run_gradcam(genus_models, "genus", cropped_images, transformations)
