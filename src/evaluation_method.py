"""
Method that takes in a user input image or set of user images and runs them through 
the loaded trained models and creates a combined classification output
"""
import sys
import os
import json
import torch
import dill
import globals
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# pylint: disable=too-many-arguments, too-many-positional-arguments
class EvaluationMethod:
    """
    Takes image input and creates a classification by running the image through
    loaded CNN models
    """

    def __init__(self, height_filename, models_dict, eval_method,
                 species_filename, accuracies_filename=None):
        """
        Load the trained models for usage and have the class prepared for user input.
        During testing phases, determining which evaluation method defined below will 
        be chosen here as well
        """
        self.use_method = eval_method     #1 = heaviest, 2 = weighted, 3 = stacked

        self.accuracies_filename = accuracies_filename

        self.trained_models = models_dict

        self.species_idx_dict = self.open_class_dictionary(species_filename)

        self.height = None
        with open(height_filename, 'r', encoding='utf-8') as file:
            self.height = int(file.readline().strip())

        #load transformations to a list for use in the program
        self.transformations = self.get_transformations()

        # initialize the size of how many classifications you want outputted by the evaluation
        self.k = 5

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

    def evaluate_image(self, late=None, dors=None, fron=None, caud=None, ood_check=False):
        """
        Create an evaluation of the input image(s) by running each given image through
        its respective model and then run the output of the models through the evaluation method
        and returns the top classifications

        Args:
            late, dors, fron, caud (PIL.Image, optional): Input images for each view.
            ood_check (bool): Whether to apply OOD detection for out-of-distribution rejection.

        Returns: List of tuples [(species_name, confidence_score), ...]
            sorted by confidence(index 0 being the highest).
            A return of None, -1 indicates an error
        """
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_built() else 'cpu')

        inputs = {
            "caud": (caud, self.transformations[0]),
            "dors": (dors, self.transformations[1]),
            "fron": (fron, self.transformations[2]),
            "late": (late, self.transformations[3]),
        }

        # Define variables outside the if statements so they can be used in other method calls
        predictions = {
            "late" : {"scores" : None, "species" : None},
            "dors" : {"scores" : None, "species" : None},
            "fron" : {"scores" : None, "species" : None},
            "caud" : {"scores" : None, "species" : None},
        }
        view_count = 0

        for view, (image, transform) in inputs.items():
            if image:
                view_count += 1
                transformed_image = self.transform_input(image, transform).to(device)

                with torch.no_grad():
                    model_output = self.trained_models[view].to(device)(transformed_image)

                if ood_check:
                    # Apply OOD for out-of-distribution detection
                    # Threshold to be adjusted (If threshold is too strict (try −14) If too lenient (try −10))
                    # _ is the energy score returned, which is not used
                    is_confident, _, softmax_scores = self.apply_ood(
                        model_output, temperature=1000, threshold=-12.0
                    )
                else:
                    softmax_scores = torch.nn.functional.softmax(model_output[0], dim=0)
                    is_confident = True  # Treat as in-distribution

                if not is_confident:
                    # Get the predicted top 5 species(or less if not enough outputs) and their indices
                    topk = min(self.k - 1, softmax_scores.size(0))
                    top_scores, top_species = torch.topk(softmax_scores, topk)
                    # Store unknown and top 4 confidences and species as a list to the correct dictionary entry
                    # Index 0(unknown) is the highest and 4 is the lowest
                    predictions[view]["scores"] = [0.0] + top_scores.tolist()
                    predictions[view]["species"] = [-1] + top_species.tolist()  # -1 means unknown
                else:
                    # Get the predicted top 5 species(or less if not enough outputs) and their indices
                    topk = min(self.k, softmax_scores.size(0))
                    top_scores, top_species = torch.topk(softmax_scores, topk)
                    # Store top 5 confidence and species as a list to the correct dictionary entry
                    # Index 0 is the highest and 4 is the lowest
                    predictions[view]["scores"] = top_scores.tolist()
                    predictions[view]["species"] = top_species.tolist()

        return self.evaluation_handler(predictions, view_count)

    def evaluation_handler(self, predictions, view_count):
        """
        Creates an evaluation by taking the predictions from the models and creating two
        nested lists of each angle and their top scores and species. With these lists
        created and the view count the method correctly calls the desired evaluation
        method and returns the predicted list.

        Returns: List of tuples [(species_name, confidence_score), ...]
            sorted by confidence(index 0 being the highest).
            A return of None, -1 indicates an error
        """
        # Create a nested list with each angles top scores(scores_list) and species(species_list)
        scores_list = []
        species_list = []
        for key in ["fron", "dors", "late", "caud"]:
            if predictions[key]["scores"]:
                scores_list.append(list(predictions[key]["scores"]))
            if predictions[key]["species"]:
                species_list.append(list(predictions[key]["species"]))

        if self.use_method == 1:
            return self.heaviest_helper_func(predictions)

        if self.use_method == 2:
            weights = []
            if self.accuracies_filename:
                with open(self.accuracies_filename, 'r', encoding='utf-8') as f:
                    accuracy_dict = json.load(f)

                for key in ["fron", "dors", "late", "caud"]:
                    if predictions[key]["scores"]:
                        weights.append(accuracy_dict[key])
                # adjust weight percentages by normalizing to sum to 1
                weights_sum = sum(weights)
                weights = [weight / weights_sum for weight in weights]
            else:
                weights = [0.25 for i in range(view_count)]

            return self.weighted_eval(scores_list, species_list, weights, view_count)

        if self.use_method == 3:
            return self.stacked_eval()

        return None, -1

    def heaviest_helper_func(self, predictions):
        """
        Handles preprocessing for heaviest is best function by finding the most
        accurate model of the input angles which is then passed to the heaviest
        is best method to get a return value

        Returns: List of tuples [(species_name, confidence_score), ...]
            sorted by confidence(index 0 being the highest).
            A return of None, -1 indicates an error
        """
        # Match uses the index returned from the method to decide which prediction to return
        accs = []
        use_angle = None
        if self.accuracies_filename:
            with open(self.accuracies_filename, 'r', encoding='utf-8') as f:
                accuracy_dict = json.load(f)

            acc_dict_reverse = {v:k for k, v in accuracy_dict.items()}

            for key in ["fron", "dors", "late", "caud"]:
                if predictions[key]["scores"]:
                    accs.append(accuracy_dict[key])

            if not accs:
                return [(None, 0), (None, 0), (None, 0), (None, 0), (None, 0)]

            use_angle = acc_dict_reverse[max(accs)]

        #base case if accuracies aren't found based on best model from experience
        elif predictions["dors"]["scores"] is not None:
            use_angle = "dors"
        elif predictions["caud"]["scores"] is not None:
            use_angle = "caud"
        elif predictions["late"]["scores"] is not None:
            use_angle = "late"
        elif predictions["fron"]["scores"] is not None:
            use_angle = "fron"

        return self.heaviest_is_best(predictions, use_angle)

    def heaviest_is_best(self, predictions, use_angle):
        """
        Takes the certainties of the models and returns the top 5 most certain predictions
        from the models based on which scores were the highest throughout the 4 models.

        Returns: List of tuples [(species_name, confidence_score), ...]
            sorted by confidence(index 0 being the highest).
        """
        top_species_scores = {}

        for i in range(0, 5):
            species_idx = predictions[use_angle]["species"][i]
            score = predictions[use_angle]["scores"][i]
            top_species_scores[species_idx] = score

        # Convert to list of (name, score)
        top_5 = []
        for key, value in top_species_scores.items():
            if key == -1 or key not in self.species_idx_dict:
                top_5.append(("Unknown Species", value))
            else:
                top_5.append((self.species_idx_dict[key], value))

        # Sort with "Unknown Species" first, then by confidence descending
        sorted_top_5 = sorted(top_5, key=lambda item: (item[0] != "Unknown Species", -item[1]))

        return sorted_top_5


    def weighted_eval(self, conf_scores, species_predictions, weights, view_count):
        """
        Takes the classifications of the models and combines them based on the normalized 
        weights from the programmer determined weights to create a list of tuples containing
        the top 5 species(from the weighted algorithm)

        Returns: List of tuples [(species_name, confidence_score), ...]
            sorted by confidence(index 0 being the highest).
        """

        top_species_scores = {}
        # Iterate through each model and perform the weighted algorithm on their top scores
        for i in range(view_count):
            if species_predictions[i] is not None:
                for rank in range(self.k):
                    species_idx = species_predictions[i][rank]
                    weighted_score = weights[i] * conf_scores[i][rank]

                    if species_idx in top_species_scores:
                        top_species_scores[species_idx] += weighted_score

                    else:
                        top_species_scores[species_idx] = weighted_score

        # Create sorted list using sorted method (list with tuples nested inside(key, value))
        sorted_scores = sorted(top_species_scores.items(), key=lambda item: item[1], reverse=True)
        # Change key from index to correct species name
        top_5 = []
        for key, value in sorted_scores:
            if key == -1 or key not in self.species_idx_dict:
                top_5.append(("Unknown Species", value))
            else:
                top_5.append((self.species_idx_dict[key], value))

        return top_5

    def enable_dropout(self, view):
        """
        Enable dropout layers for MC Dropout inference while keeping
        other layers (e.g., BatchNorm, Linear) in eval mode.
        """
        for m in self.trained_models[view].modules():
            if isinstance(m, torch.nn.Dropout) or m.__class__.__name__.startswith('Dropout'):
                m.train()

    def evaluate_image_mc_dropout(self, late=None, dors=None, fron=None, caud=None,
                                n_samples=20):
        """
        Create an evaluation of the input image(s) using Monte Carlo Dropout only.
        This method performs multiple stochastic forward passes with dropout enabled,
        then averages predictions and computes predictive uncertainty.

        Args:
            late, dors, fron, caud (PIL.Image, optional): Input images for each view.
            n_samples (int): Number of stochastic forward passes.

        Returns:
            dict: Dictionary with per-view predictions including:
                - "mean_scores": top-k averaged softmax probabilities
                - "species": top-k predicted species indices
                - "uncertainty": predictive entropy (higher = more uncertain)
        """
        device = torch.device('cuda' if torch.cuda.is_available()
                            else 'mps' if torch.backends.mps.is_built() else 'cpu')

        inputs = {
            "caud": (caud, self.transformations[0]),
            "dors": (dors, self.transformations[1]),
            "fron": (fron, self.transformations[2]),
            "late": (late, self.transformations[3]),
        }

        predictions = {}
        for view, (image, transform) in inputs.items():
            if not image:
                continue

            transformed_image = self.transform_input(image, transform).to(device)

            # keep model in eval but activate dropout
            self.trained_models[view].eval()
            self.enable_dropout(view)

            # collect predictions across n_samples passes
            softmax_samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    logits = self.trained_models[view](transformed_image)
                    softmax_probs = torch.nn.functional.softmax(logits, dim=1)
                    softmax_samples.append(softmax_probs.cpu())

            # stack into [n_samples, num_classes]
            softmax_samples = torch.stack(softmax_samples)
            mean_probs = softmax_samples.mean(dim=0)[0]  # [num_classes]
            mean_probs = mean_probs / mean_probs.sum()  # ensure proper normalization
            entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum().item()

            # Top-k predictions
            topk = min(self.k, mean_probs.size(0))
            top_scores, top_species = torch.topk(mean_probs, topk)

            # Map indices to species names
            mapped_species = []
            for idx in top_species.tolist():
                if idx == -1 or idx not in self.species_idx_dict:
                    mapped_species.append("Unknown Species")
                else:
                    mapped_species.append(self.species_idx_dict[idx])

            predictions[view] = {
                "mean_scores": top_scores.tolist(),
                "species": mapped_species,
                "uncertainty": entropy
            }

            self.trained_models[view].eval()  # reset back to eval after MC dropout passes

        return predictions

    def evaluate_heaviest_mc_dropout(self, late=None, dors=None, fron=None, caud=None,
                                            n_samples=20, uncertainty_threshold=0.8):
        """
        Stepwise MC Dropout evaluation that starts with the best model (highest accuracy)
        and moves down the list if uncertainty exceeds the threshold.

        If no model passes, returns rejection info with best model's uncertainty and scores.
        """
        device = torch.device('cuda' if torch.cuda.is_available()
                            else 'mps' if torch.backends.mps.is_built() else 'cpu')

        inputs = {
            "caud": (caud, self.transformations[0]),
            "dors": (dors, self.transformations[1]),
            "fron": (fron, self.transformations[2]),
            "late": (late, self.transformations[3]),
        }

        # Load accuracy dictionary and rank views (highest to lowest)
        with open(self.accuracies_filename, 'r', encoding='utf-8') as f:
            accuracy_dict = json.load(f)
        ranked_views = sorted(accuracy_dict.keys(), key=lambda k: accuracy_dict[k], reverse=True)

        best_result = None
        for view in ranked_views:
            image, transform = inputs.get(view, (None, None))
            if image is None:
                continue

            transformed_image = self.transform_input(image, transform).to(device)

            # Activate dropout layers
            self.trained_models[view].eval()
            self.enable_dropout(view)

            # Collect MC Dropout samples
            softmax_samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    logits = self.trained_models[view](transformed_image)
                    softmax_probs = torch.nn.functional.softmax(logits, dim=1)
                    softmax_samples.append(softmax_probs.cpu())

            softmax_samples = torch.stack(softmax_samples)
            mean_probs = softmax_samples.mean(dim=0)[0]
            mean_probs = mean_probs / mean_probs.sum()  # ensure normalization
            entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum().item()

            # Top-k predictions
            topk = min(self.k, mean_probs.size(0))
            top_scores, top_indices = torch.topk(mean_probs, topk)

            # Map species names
            species_names = [
                self.species_idx_dict.get(idx, "Unknown Species")
                for idx in top_indices.tolist()
            ]

            result = {
                "view": view,
                "mean_scores": top_scores.tolist(),
                "species": species_names,
                "uncertainty": entropy,
            }

            # Save best (first) result for fallback
            if best_result is None:
                best_result = result

            # Threshold check
            if entropy < uncertainty_threshold:
                result["status"] = "accepted"
                return result

            self.trained_models[view].eval()  # reset to eval after MC dropout passes

        # Fallback: reject if all models uncertain
        best_result["status"] = "rejected"
        return best_result

    def stacked_eval(self):
        """
        Takes the classifications of the models and runs them through another model that determines
        the overall output

        REACH CASE/STUB FOR SPRINT 3

        Returns: classification of combined models
        """

    def transform_input(self, image_input, transformation):
        """
        Takes the app side's image and a given transformation
        and transforms it to fit our model

        Returns: transformed image for classification
        """
        transformed_image = transformation(image_input)
        transformed_image = transformed_image.unsqueeze(0)

        return transformed_image

    def apply_ood(self, logits, temperature=1000.0, threshold=-10.0):
        """
        Applies OOD (out-of-distribution detection) using energy scores.

        Args:
            logits (Tensor): Raw model outputs.
            temperature (float): Temperature for scaling.
            threshold (float): Energy threshold for rejection.

        Returns:
            Tuple[bool, float, Tensor]: 
                - is_confident (bool): True if in-distribution, False if likely OOD.
                - energy_score (float)
                - softmax_probs (Tensor)
        """
        scaled_logits = logits / temperature
        softmax_probs = torch.nn.functional.softmax(scaled_logits, dim=1)
        energy_score = -temperature * torch.logsumexp(scaled_logits, dim=1)
        is_confident = energy_score.item() < threshold  # lower = less confident
        return is_confident, energy_score.item(), softmax_probs[0]
