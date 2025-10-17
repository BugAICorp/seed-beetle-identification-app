# flake8: noqa
"""
Method that takes in a user input image or set of user images and runs them through 
the loaded trained models and creates a combined classification output
"""
import sys
import os
import json
import torch
import dill
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class GenusEvaluationMethod:
    """
    Takes image input and creates a classification by running the image through
    loaded CNN models
    """

    def __init__(self, height_filename, models_dict, eval_method,
                 genus_filename, accuracies_filename=None):
        """
        Load the trained models for usage and have the class prepared for user input.
        During testing phases, determining which evaluation method defined below will 
        be chosen here as well
        """
        self.use_method = eval_method     #1 = heaviest, 2 = weighted, 3 = stacked

        self.accuracies_filename = accuracies_filename

        self.trained_models = models_dict

        self.genus_idx_dict = self.open_class_dictionary(genus_filename)

        self.height = None
        with open(height_filename, 'r', encoding='utf-8') as file:
            self.height = int(file.readline().strip())

        #load transformations to a list for use in the program
        self.transformations = self.get_transformations()

    def open_class_dictionary(self, filename):
        """
        Open and save the class dictionary for use in the evaluation method 
        to convert the model's index to a string genus classification

        Returns: dictionary defined by file
        """
        with open(filename, 'r', encoding='utf-8') as json_file:
            class_dict_read = json.load(json_file)

        #Convert string keys to integers(keys automatically switched by json save)
        #Undoes issues created by json saving
        class_dict = {}

        for key, value in class_dict_read.items():
            class_dict[int(key)] = value

        return class_dict

    def get_transformations(self):
        """
        Create and return a list of transformations for each angle using
        the pre-made transformation files

        Returns: list of transformations
        """
        transformations = []

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/caud_transformation.pth"), "rb") as f:
            transformations.append(dill.load(f))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/dors_transformation.pth"), "rb") as f:
            transformations.append(dill.load(f))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/fron_transformation.pth"), "rb") as f:
            transformations.append(dill.load(f))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/late_transformation.pth"), "rb") as f:
            transformations.append(dill.load(f))

        return transformations

    def evaluate_image(self, late=None, dors=None, fron=None, caud=None, ood_check = False):
        """
        Create an evaluation of the input image(s) by running each given image through
        its respective model and then run the output of the models through the evaluation method
        and return the classification

        Args:
            late, dors, fron, caud (PIL.Image, optional): Input images for each view.
            ood_check (bool): Whether to apply OOD detection for out-of-distribution rejection.

        Returns: Classification of input images and confidence score. 
                A return of None, -1 indicates an error
        """
        # Set device to a CUDA-compatible gpu
        # Else use CPU to allow general usability and MPS if user has Apple Silicon
        device = torch.device(
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_built()
            else 'cpu')

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

                if ood_check:
                    # Apply OOD for out-of-distribution detection
                    # Threshold to be adjusted (If threshold is too strict (try −14) If too lenient (try −10))
                    # energy is not needed but is returned
                    is_confident, _, softmax_scores = self.apply_ood(
                        model_output, temperature=1000, threshold=-12.0
                    )
                else:
                    softmax_scores = torch.nn.functional.softmax(model_output[0], dim=0)
                    is_confident = True  # Treat as in-distribution

                if is_confident:
                    # Use the predicted class and softmax confidence
                    confidence, predicted_index = torch.max(softmax_scores, 0)
                    predictions[view]["score"] = confidence.item()
                    predictions[view]["genus"] = predicted_index.item()
                else:
                    # Mark as unknown if not confident
                    predictions[view]["score"] = 0.0
                    predictions[view]["genus"] = -1  # -1 indicates unknown class

        return self.evaluation_handler(predictions, view_count)

    def evaluation_handler(self, predictions, view_count):
        """
        Creates an evaluation by taking the predictions from the models and creating two
        nested lists of each angle and their top scores and genera. With these lists
        created and the view count the method correctly calls the desired evaluation
        method and returns a prediction tuple.

        Returns: tuple (genus_name, confidence_score)
            A return of None, -1 indicates an error
        """

        if self.use_method == 1:
            #match uses the index returned from the method to decide which prediction to return
            return self.heaviest_is_best([predictions["fron"]["score"],
                                       predictions["dors"]["score"],
                                       predictions["late"]["score"],
                                       predictions["caud"]["score"]],
                                      [predictions["fron"]["genus"],
                                       predictions["dors"]["genus"],
                                       predictions["late"]["genus"],
                                       predictions["caud"]["genus"]])

        if self.use_method == 2:
            # Initialize weights for use in weighted eval, using the genus models accuracies
            weights = []
            score_list = []
            genus_list = []
            if self.accuracies_filename:
                with open(self.accuracies_filename, 'r', encoding='utf-8') as f:
                    accuracy_dict = json.load(f)

                for key in ["fron", "dors", "late", "caud"]:
                    if predictions[key]["genus"]:
                        weights.append(accuracy_dict[key])
                        score_list.append(predictions[key]["score"])
                        genus_list.append(predictions[key]["genus"])
                # adjust weight percentages by normalizing to sum to 1
                weights_sum = sum(weights)
                weights = [weight / weights_sum for weight in weights]
            else:
                weights = [0.25 for i in range(view_count)]

            return self.weighted_eval(score_list, genus_list, weights, view_count)

        if self.use_method == 3:
            return self.stacked_eval()

        return None, -1

    def heaviest_is_best(self, conf_scores, genus_predictions):
        """
        Takes the certainties of the models and returns the most 
        certain model's specification

        Returns: specifies most certain model
        """
        view_order = ["fron", "dors", "late", "caud"]

        if self.accuracies_filename:
            with open(self.accuracies_filename, 'r', encoding='utf-8') as f:
                accuracy_dict = json.load(f)

            # Only consider views with input   
            valid_views = []
            for i, view in enumerate(view_order):
                if genus_predictions[i] is not None:
                    # stored as a tuple containing view, index to prediction, model accuracy from dict
                    valid_views.append((view, i, accuracy_dict[view]))

            if not valid_views:
                return None, -1

            # Pick view with highest accuracy
            best_view, best_index, _ = max(valid_views, key=lambda x: x[2])

        else:
            # Fallback priority order
            priority = ["dors", "late", "caud", "fron"]
            for view in priority:
                idx = view_order.index(view)
                if genus_predictions[idx] is not None:
                    best_index = idx
                    break
            else:
                return None, -1

        genus = genus_predictions[best_index]
        score = conf_scores[best_index]

        if genus == -1 or genus not in self.genus_idx_dict:
            return "Unknown Genus", score

        return self.genus_idx_dict[genus], score


    def weighted_eval(self, conf_scores, genus_predictions, weights, view_count):
        """
        Takes the classifications of the models and combines them based on programmer determined
        weights to create a single output, ignoring OOD (unknown) predictions.

        Returns: classification of combined models
        """

        genus_scores = {}
        for i in range(view_count):
            genus = genus_predictions[i]
            if genus == -1 or genus is None:
                # Skip so weighted average isn't skewed
                continue

            weighted_score = weights[i] * conf_scores[i]
            if genus in genus_scores:
                genus_scores[genus] += weighted_score
            else:
                genus_scores[genus] = weighted_score

        if not genus_scores:
            # If no valid genus predictions, return unknown
            return "Unknown Genus", 0.0

        highest_genus = max(genus_scores, key=genus_scores.get)
        highest_score = genus_scores[highest_genus]

        return self.genus_idx_dict.get(highest_genus, "Unknown Species"), highest_score

    def enable_dropout(self, view):
        """
        Enable dropout layers for MC Dropout inference while keeping
        other layers (e.g., BatchNorm, Linear) in eval mode.
        """
        for m in self.trained_models[view].modules():
            if isinstance(m, torch.nn.Dropout) or m.__class__.__name__.startswith('Dropout'):
                m.train()

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

            # Top-1 prediction
            score, genus_idx = torch.max(mean_probs, dim=0)
            genus_name = self.genus_idx_dict.get(genus_idx.item(), "Unknown Genus")

            result = {
                "view": view,
                "mean_score": score.item(),
                "genus": genus_name,
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
        Takes the app side's image and transforms it to fit our model

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
        # lower energy = in-distribution (more confident)
        is_confident = energy_score.item() > threshold
        return is_confident, energy_score.item(), softmax_probs[0]
