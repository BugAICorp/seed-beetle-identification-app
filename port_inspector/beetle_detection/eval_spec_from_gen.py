"""eval_species_by_genus.py"""
import sys
import os
import json
import torch
import dill
import globals
from model_loader import load_genus_specific_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# pylint: disable=too-many-arguments, too-many-positional-arguments
class EvalSpeciesByGenus:
    """
    Takes image input and evaluate species and genus. Uses genus eval to specify which species
    models to use when evaluating
    """

    def __init__(self, models_dict, genus_filename):
        self.trained_models = models_dict
        self.species_model = None
        self.species_idx_dict = None
        self.genus_idx_dict = self.open_class_dictionary(genus_filename)

        #load transformations to a list for use in the program
        self.transformations = self.get_transformations()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        # initialize the size of how many classifications you want outputted by the evaluation
        self.k = 5

    def classify_images(self, caud=None, dors=None, fron=None, late=None):
        """
        Handle classification of both genus and species and return both of
        their classification in their proper formats
        """
        genus_class, genus_score = self.get_genus(caud=caud, dors=dors, fron=fron, late=late)
        print(f"{genus_class}")

        self.load_species_models(genus_class)
        if self.species_model is None or self.species_idx_dict is None:
            return (None, 0), [(None, 0)]

        species_class = self.get_species(caud=caud, dors=dors, fron=fron, late=late)

        return (genus_class, genus_score), species_class

    def open_class_dictionary(self, filename):
        """
        Open and save the class dictionary for use in the evaluation method 
        to convert the model's index to a string species classification

        Returns: dictionary defined by file
        """
        with open(filename, 'r', encoding='utf-8') as json_file:
            class_dict = json.load(json_file)

        # Convert string keys to integers(because of how the dictionary was saved with json)
        class_dict = {int(key): value for key, value in class_dict.items()}

        return class_dict

    def get_transformations(self):
        """
        Create and return a list of transformations for each angle using
        the pre-made transformation files

        Returns: list of transformations
        """
        transformations = []

        #open each file and load the transformation then save it to the list
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/caud_transformation.pth"), "rb") as f:
            transformations.append(dill.load(f))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/dors_transformation.pth"), "rb") as f:
            transformations.append(dill.load(f))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/fron_transformation.pth"), "rb") as f:
            transformations.append(dill.load(f))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/late_transformation.pth"), "rb") as f:
            transformations.append(dill.load(f))

        return transformations

    def get_genus(self, caud=None, dors=None, fron=None, late=None):
        """
        Return the top genus evaluated by the genus models
        """
        # Set device to a CUDA-compatible gpu
        # Else use CPU to allow general usability and MPS if user has Apple Silicon
        device = self.device
        inputs = {
            "caud": (caud, self.transformations[0]),
            "dors": (dors, self.transformations[1]),
            "fron": (fron, self.transformations[2]),
            "late": (late, self.transformations[3]),
        }

        # Define variables outside the if statements so they can be used in other method calls
        predictions = {
            "late" : {"score" : 0, "genus" : None},
            "dors" : {"score" : 0, "genus" : None},
            "fron" : {"score" : 0, "genus" : None},
            "caud" : {"score" : 0, "genus" : None},
        }
        view_count = 0

        for view, (image, transformation) in inputs.items():
            if image:
                view_count += 1
                transformed_image = self.transform_input(image, transformation).to(device)

                with torch.no_grad():
                    model_output = self.trained_models[view].to(device)(transformed_image)

                # Get the predicted class and confidence score
                _, predicted_index = torch.max(model_output, 1)
                predictions[view]["score"] = torch.nn.functional.softmax(
                    model_output, dim=1)[0, predicted_index].item()
                predictions[view]["genus"] = predicted_index.item()

        certainties = []
        genera = []
        for key in ["caud", "dors", "fron", "late"]:
            certainties.append(predictions[key]["score"])
            genera.append(predictions[key]["genus"])

        i = certainties.index(max(certainties))
        return self.genus_idx_dict[genera[i]], certainties[i]

    # pylint: disable=too-many-branches
    def get_species(self, caud=None, dors=None, fron=None, late=None):
        """
        Get the species classification based on the loaded species models
        """
        scores = []
        species = []
        device = self.device

        all_inputs = []
        input_order = []
        if caud:
            all_inputs.append(caud)
            input_order.append(0)
        if dors:
            all_inputs.append(dors)
            input_order.append(1)
        if fron:
            all_inputs.append(fron)
            input_order.append(2)
        if late:
            all_inputs.append(late)
            input_order.append(3)

        #Check if there are less possible classifications than self.k and adjust if necessary
        k = min(self.k, len(self.species_idx_dict))

        count = 0
        for i in all_inputs:
            image = self.transform_input(i, self.transformations[input_order[count]]).to(device)

            with torch.no_grad():
                output = self.species_model.to(device)(image)

            # Get the predicted top 5 species(or less if not enough outputs) and their indices
            softmax_scores = torch.nn.functional.softmax(output, dim=1)[0]
            top5_scores, top5_species = torch.topk(softmax_scores, k)
            print(top5_species)

            # Store top 5 confidence and species as a list to the correct dictionary entry
            # Index 0 is the highest and 4 is the lowest
            scores.append(top5_scores.tolist())
            species.append(top5_species.tolist())
            count += 1

        top_five_scores = []
        top_five_names = []
        for i in range(k):
            top_five_scores.append(0)
            top_five_names.append(None)

        for i, _ in enumerate(species):
            for j in range(len(species[i])):
                if species[i][j] in top_five_names:
                    species_index = top_five_names.index(species[i][j])
                    top_five_scores[species_index] = max(top_five_scores[species_index], scores[i][j])
                elif scores[i][j] > min(top_five_scores):
                    index_of_lowest = top_five_scores.index(min(top_five_scores))
                    top_five_scores[index_of_lowest] = scores[i][j]
                    top_five_names[index_of_lowest] = species[i][j]

        top_classes = {}
        for i, _ in enumerate(top_five_scores):
            top_classes[top_five_names[i]] = top_five_scores[i]
        sorted_scores = sorted(top_classes.items(), key=lambda item: item[1], reverse=True)

        list_to_return = []
        for key, value in sorted_scores:
            if key in self.species_idx_dict:
                list_to_return.append((self.species_idx_dict[key], value))
            else:
                list_to_return.append(("Unknown Species", value))

        while len(list_to_return) < 5:
            list_to_return.append(("No other species", 0))

        return list_to_return

    # pylint: enable=too-many-branches
    def load_species_models(self, genus):
        """
        Load the species classification model based on the determined
        genus
        """
        self.species_model, self.species_idx_dict = load_genus_specific_model(genus, self.device)

    def transform_input(self, image_input, transformation):
        """
        Takes the app side's image and transforms it to fit our model

        Returns: transformed image for classification
        """
        transformed_image = transformation(image_input)
        transformed_image = transformed_image.unsqueeze(0)

        return transformed_image
