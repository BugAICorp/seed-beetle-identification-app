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

    def __init__(self, height_filename, models_dict, genus_filename):
        self.trained_models = models_dict
        self.species_model = None
        self.species_idx_dict = None
        self.genus_idx_dict = self.open_class_dictionary(genus_filename)
        
        self.height = None
        with open(height_filename, 'r', encoding='utf-8') as file:
            self.height = int(file.readline().strip())

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
        with open(globals.caud_transformation, "rb") as f:
            transformations.append(dill.load(f))

        with open(globals.dors_transformation, "rb") as f:
            transformations.append(dill.load(f))

        with open(globals.fron_transformation, "rb") as f:
            transformations.append(dill.load(f))

        with open(globals.late_transformation, "rb") as f:
            transformations.append(dill.load(f))

        return transformations

    def get_genus(self, caud=None, dors=None, fron=None, late=None):
        """
        Return the top genus evaluated by the genus models
        """
        device = self.device

        #define variables outside the if statements so they can be used in other method calls
        predictions = {
            "late" : {"score" : 0, "genus" : None},
            "dors" : {"score" : 0, "genus" : None},
            "fron" : {"score" : 0, "genus" : None},
            "caud" : {"score" : 0, "genus" : None},
        }

        view_count = 0

        if late:
            view_count += 1
            late_image = self.transform_input(late, self.transformations[3]).to(device)

            with torch.no_grad():
                late_output = self.trained_models["late"].to(device)(late_image)

            # Get the predicted class and confidence score
            _, predicted_index = torch.max(late_output, 1)
            predictions["late"]["score"] = torch.nn.functional.softmax(
                late_output, dim=1)[0, predicted_index].item()
            predictions["late"]["genus"] = predicted_index.item()

        if dors:
            view_count += 1
            #mirrors above usage but for the dors angle
            dors_image = self.transform_input(dors, self.transformations[1]).to(device)

            with torch.no_grad():
                dors_output = self.trained_models["dors"].to(device)(dors_image)

            _, predicted_index = torch.max(dors_output, 1)
            predictions["dors"]["score"] = torch.nn.functional.softmax(
                dors_output, dim=1)[0, predicted_index].item()
            predictions["dors"]["genus"] = predicted_index.item()

        if fron:
            view_count += 1
            #mirrors above usage but for the fron angle
            fron_image = self.transform_input(fron, self.transformations[2]).to(device)

            with torch.no_grad():
                fron_output = self.trained_models["fron"].to(device)(fron_image)

            _, predicted_index = torch.max(fron_output, 1)
            predictions["fron"]["score"] = torch.nn.functional.softmax(
                fron_output, dim=1)[0, predicted_index].item()
            predictions["fron"]["genus"] = predicted_index.item()

        if caud:
            view_count += 1
            #mirrors above usage but for the caud angle
            caud_image = self.transform_input(caud, self.transformations[0]).to(device)

            with torch.no_grad():
                caud_output = self.trained_models["caud"].to(device)(caud_image)

            _, predicted_index = torch.max(caud_output, 1)
            predictions["caud"]["score"] = torch.nn.functional.softmax(
                caud_output, dim=1)[0, predicted_index].item()
            predictions["caud"]["genus"] = predicted_index.item()

        certainties = []
        genera = []
        for key in ["caud", "dors", "fron", "late"]:
            certainties.append(predictions[key]["score"])
            genera.append(predictions[key]["genus"])

        i = certainties.index(max(certainties))
        return self.genus_idx_dict[genera[i]], certainties[i]

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

        count = 0
        for i in all_inputs:
            image = self.transform_input(i, self.transformations[input_order[count]]).to(device)

            with torch.no_grad():
                output = self.species_model.to(device)(image)

            # Get the predicted top 5 species(or less if not enough outputs) and their indices
            softmax_scores = torch.nn.functional.softmax(output, dim=1)[0]
            top5_scores, top5_species = torch.topk(softmax_scores, self.k)

            # Store top 5 confidence and species as a list to the correct dictionary entry
            # Index 0 is the highest and 4 is the lowest
            scores.append(top5_scores.tolist())
            species.append(top5_species.tolist())
            count += 1

        top_five_scores = [0, 0, 0, 0, 0]
        top_five_names = [None, None, None, None, None]
        for i in range(len(species)):
            for j in range(len(species[i])):
                if species[i][j] in top_five_names:
                    species_index = top_five_names.index(species[i][j])
                    top_five_scores[species_index] = max(top_five_scores[species_index], scores[i][j])
                elif (scores[i][j] > min(top_five_scores)):
                    index_of_lowest = top_five_scores.index(min(top_five_scores))
                    top_five_scores[index_of_lowest] = scores[i][j]
                    top_five_names[index_of_lowest] = species[i][j]

        top_classes = {}
        for i in range(len(top_five_scores)):
            top_classes[top_five_names[i]] = top_five_scores[i]
        sorted_scores = sorted(top_classes.items(), key=lambda item: item[1], reverse=True)

        list_to_return = []
        for key, value in sorted_scores:
            if key in self.species_idx_dict:
                list_to_return.append((self.species_idx_dict[key], value))
            else:
                list_to_return.append(("Unknown Species", value))

        return list_to_return

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
