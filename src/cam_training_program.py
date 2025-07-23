""" cam_training_program.py """

import os
import sys
import json
import copy
import gc
from io import BytesIO
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import dill
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformation_classes import HistogramEqualization

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class GradCAM:
    """
    Implements Grad-CAM for a given model and target layer.
    Used to generate class activation heatmaps for model interpretation.
    """
    def __init__(self, model, target_layer_module):
        """ 
        Initialize Grad-CAM with model and target layer. 
        
        Args:
            model(torch.nn.Module): The neural network model (ResNet50)
            target_layer(str):The name of the layer inside the model from
                which Grad-CAM will compute the activations.
        """
        self.model = model
        self.model.to(next(model.parameters()).device)
        self.target_layer = target_layer_module
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """ Registers forward and backward hooks to capture activations and gradients. """
        def forward_hook(_module, _input, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates Grad-CAM heatmap.

        Args:
            input_tensor (torch.Tensor): Input images.
            class_idx (list[int], optional): Class indices to guide backpropagation.

        Returns:
            torch.Tensor: Normalized heatmaps of shape [B, H, W].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Ensure gradients can be computed
        if not output.requires_grad:
            output.requires_grad_(True)

        # Scalar backward: avoids issues with hooks not firing
        if class_idx is None:
            class_idx = output.argmax(dim=1)

        target = output[range(output.size(0)), class_idx]
        target.sum().backward(retain_graph=True)

        if self.gradients is None:
            raise RuntimeError(
                "Gradients not captured. Check if backward hook is registered and .backward() was called.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam_norm

    def remove_hooks(self):
        """Remove all hooks after Grad-CAM usage."""
        for handle in self.hook_handles:
            handle.remove()

# pylint: disable=too-many-instance-attributes
class CAMGuidedTrainingProgram:
    """
    Custom training pipeline that incorporates CAM-guided supervision
    to align model attention with human-provided attention masks.
    """
    def __init__(self, dataframe, class_column, num_classes, mask_dir, image_column='Image'):
        """
        Initializes the CAM-guided training program.

        Args:
            dataframe (pd.DataFrame): Full dataset containing image binaries and labels.
            class_column (str): Column name for class labels in the dataframe.
            num_classes (int): Total number of unique classes in the classification task.
            mask_dir (str): Directory where binary attention mask images are stored.
            image_column (str, optional): Column name for image binary data. Defaults to 'Image'.
        """
        self.dataframe = dataframe
        if "Filename" not in self.dataframe.columns:
            raise ValueError("Missing 'Filename' column in dataframe — required for mask loading.")
        self.height = 300
        self.num_classes = num_classes
        # Dataframe variables
        self.image_column = image_column
        self.class_column = class_column
        # subsets to save database reading to
        self.subsets = {
            "caud" : self.get_subset("CAUD", self.dataframe),
            "dors" : self.get_subset("DORS", self.dataframe),
            "fron" : self.get_subset("FRON", self.dataframe),
            "late" : self.get_subset("LATE", self.dataframe)
        }
        # Set device to a CUDA-compatible gpu, else mps, else cpu
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_built()
            else 'cpu')
        self.models = {
            "caud" : self.load_model(),
            "dors" : self.load_model(),
            "fron" : self.load_model(),
            "late" : self.load_model()
        }
        # Dictionary variables
        self.class_index_dictionary = {}
        self.class_string_dictionary = {}
        self.class_set = set()
        # Model accuracy dictionary
        self.model_accuracies = {
            "caud" : 0,
            "dors" : 0,
            "fron" : 0,
            "late" : 0
        }

        classes = dataframe[self.class_column].values
        unique_classes = sorted(set(classes))
        self.class_string_dictionary = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.class_index_dictionary = {idx: cls for cls, idx in self.class_string_dictionary.items()}
        self.class_set = set(self.class_string_dictionary.values())

        # Create transformation method dictionary
        self.transformations = {
            "caud": transforms.Compose([
        transforms.Resize((self.height, self.height)),
        transforms.ToTensor(),
        HistogramEqualization(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            "dors": transforms.Compose([
        transforms.Resize((self.height, self.height)),
        transforms.ToTensor(),
        HistogramEqualization(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            "fron": transforms.Compose([
        transforms.Resize((self.height, self.height)),
        transforms.ToTensor(),
        HistogramEqualization(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            "late": transforms.Compose([
        transforms.Resize((self.height, self.height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        }

        self.train_transformations = self.create_train_transformations(
            rotation_degree=5,brightness=0.1, contrast=0.1, erasing=(0.5, (0.02, 0.15)))

        self.mask_dir = mask_dir
        self.lambda_attn = 0.1

    def get_subset(self, view_type, dataframe):
        """
        Reads database and pulls subset where View column is equal to parameter, view_type
        
        Args:
            view_type (string): View type column value (e.g., 'CAUD', 'DORS', 'FRON', 'LATE')
       
        Return: 
            pd.DataFrame: Subset of database if column value valid, otherwise empty dataframe
        """
        return dataframe[dataframe["View"] == view_type] if not dataframe.empty else pd.DataFrame()

    def create_train_transformations(
            self, rotation_degree=5, brightness=0.1, contrast=0.1, erasing=(0.5, (0.02, 0.15))):
        """
        Takes the self.transformations dictionary and forms training transformations. This allows for
        data augmention while training(rotation, noise, etc.). This transformation contains random rotation,
        random brightness and contrast adjustments, and random pixel erasing.

        Args:
            rotation_degree (int): Maximum degree of random rotation applied to training images.
            brightness (float): Maximum brightness jitter factor; the image brightness is adjusted.
            contrast (float): Maximum contrast jitter factor.
            erasing (tuple): A tuple (p, scale), where:
                             - p (float): Probability of applying random erasing.
                             - scale (tuple of float): Range of proportion of erased area against input image.
        
        Returns:
            dict: A dictionary of transformation pipelines with keys corresponding to their respective image views.
        """
        # Variables used for random erasing
        p = erasing[0]
        scale = erasing[1]

        train_transformations = {}
        for key in ["caud", "dors", "fron", "late"]:
            base_transforms = self.transformations[key].transforms

            # Manually reorder to insert augmentations in the correct locations
            new_transforms = []
            normalize_transform = None
            for t in base_transforms:
                if isinstance(t, transforms.Resize):
                    new_transforms.append(t)
                    # Add PIL augmentations here
                    new_transforms.append(transforms.RandomRotation(degrees=rotation_degree))
                    new_transforms.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
                    # End of PIL augmentations
                elif isinstance(t, transforms.ToTensor):
                    new_transforms.append(t)
                elif isinstance(t, HistogramEqualization):
                    new_transforms.append(t)
                elif isinstance(t, transforms.Normalize):
                    normalize_transform = t
                else:
                    new_transforms.append(t)

            # Add tensor augmentations here
            new_transforms.append(transforms.RandomErasing(p=p, scale=scale))
            # End of tensor augmentations
            if normalize_transform:
                new_transforms.append(normalize_transform)

            train_transformations[key] = transforms.Compose(new_transforms)

        return train_transformations

    def load_attention_masks(self, paths):
        """
        Loads binary masks from disk corresponding to image paths.

        Args:
            paths (list[str]): List of image file paths.

        Returns:
            torch.Tensor: Batch of binary masks [B, H, W] with values 0.0 or 1.0.
        """
        masks = []
        for img_path in paths:
            base_name = os.path.basename(img_path).replace('.jpg', '.png')
            mask_path = os.path.join(self.mask_dir, base_name)
            if not os.path.exists(mask_path):
                masks.append(torch.zeros((1, self.height, self.height)))  # no attention guidance
                continue

            mask_img = Image.open(mask_path).convert("L")  # Load as grayscale
            mask_tensor = transforms.ToTensor()(mask_img) # [1, H, W], range [0.0, 1.0]
            binary_mask = (mask_tensor > 0.5).float() # Binarize: 1.0 = beetle, 0.0 = background
            masks.append(binary_mask)

        masks = torch.stack(masks).squeeze(1)
        return masks.to(self.device)

    def get_train_test_split(self, df):
        """
        Gets train and test split for given dataframe

        Returns:
            list: [train_x, test_x, train_y, test_y, train_paths, test_paths]
        """
        image_binaries = df[self.image_column].values
        classes = df[self.class_column].values
        filenames = df["Filename"].values
        labels = [self.class_string_dictionary[label] for label in classes]
        # Split subset into training and testing sets
        # x: images, y: species
        train_x, test_x, train_y, test_y, train_paths, test_paths = train_test_split(
        image_binaries, labels, filenames, test_size=0.2)

        return [train_x, test_x, train_y, test_y, train_paths, test_paths]


    def train(self, num_epochs, train_loader, test_loader, view, lrate=0.001):
        """
        Trains the model using standard + CAM guidance loss.

        Args:
            num_epochs (int): Number of training epochs.
            train_loader (DataLoader): Training data loader.
            test_loader (DataLoader): Evaluation data loader.
        """

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.models[view].parameters(), lr=lrate)
        model = self.models[view]
        grad_cam = GradCAM(model, target_layer_module=model.layer4)

        best_epoch = 0
        best_macro_f1 = 0
        best_state_dict = None

        for epoch in range(num_epochs):
            self.models[view].train()
            running_loss = 0.0

            for inputs, labels, paths in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Clear previous gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.models[view](inputs)
                # Compute standard classification loss (CrossEntropy)
                pred_loss = criterion(outputs, labels)

                # Generate Grad-CAM heatmaps for the input batch
                cam = grad_cam.generate_heatmap(inputs)

                # Load corresponding binary attention masks from disk
                masks = self.load_attention_masks(paths)

                if masks.sum() == 0:
                    total_loss = pred_loss
                else:
                    # Compute attention alignment loss (e.g., KL divergence between CAM and mask)
                    attn_loss = self.cam_loss(cam, masks)
                    # Combine classification loss and attention loss (weighted by lambda_attn)
                    total_loss = pred_loss + self.lambda_attn * attn_loss

                # Backpropagate combined loss
                total_loss.backward()
                # Update model parameters
                optimizer.step()

                running_loss += total_loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

            macro_f1 = self.evaluate(test_loader, view)
            if macro_f1 > best_macro_f1:
                best_epoch = epoch + 1
                best_macro_f1 = macro_f1
                best_state_dict = copy.deepcopy(self.models[view].state_dict())
                print("New best model found.")
                print(f"Model accuracy improved after epoch {best_epoch}.")
            else:
                print(f"No improvement to model, the best epoch is {best_epoch}.")

        if best_state_dict is not None:
            self.models[view].load_state_dict(best_state_dict)
            self.model_accuracies[view] = best_macro_f1
            print(f"Best Macro F1: {100 * best_macro_f1:.2f}% — model loaded.")

    def evaluate(self, test_loader, view):
        """
        Evaluates the model on test set using macro F1 score.

        Args:
            test_loader (DataLoader): Evaluation data loader.

        Returns:
            float: Macro F1 score.
        """
        self.models[view].eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.models[view](inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if total != 0:
            accuracy = correct / total
            print(f"Accuracy: {100 * accuracy:.2f}%")

            # Compute and print F1 scores
            weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
            macro_f1 = f1_score(all_labels, all_predictions, average='macro')
            print(f"Weighted F1 Score: {100 * weighted_f1:.2f}%")
            print(f"Macro F1 Score: {100 * macro_f1:.2f}%")
            return macro_f1
        return None

    def train_model(self, num_epochs, view, batch, rotation=5, brightness=0.1, lrate=0.001):
        """
        Trains resnet model with subset of specified image views
        and save model to respective save file.

        Args:
            num_epochs (int): Number of epochs to train the model.
            view (str): Image view identifier (e.g., 'caud', 'dors', etc.).
            batch (int): Batch size for training and evaluation.
            rotation (int, optional): Maximum degrees of random rotation applied to training images. Default is 5.
            brightness (float, optional): Brightness jitter range for data augmentation. Default is 0.1.
            lrate (float, optional): Learning rate for optimizer. Default is 0.001.

        Returns:
            None
        """
        # Get training and testing data
        train_x, test_x, train_y, test_y, train_paths, test_paths = self.get_train_test_split(self.subsets[view])
        # Define image training transformations, placeholder for preprocessing
        self.train_transformations = self.create_train_transformations(
            rotation_degree=rotation,
            brightness=brightness,
            contrast=0.1,
            erasing=(0.5, (0.02, 0.15))
        )

        # Create DataLoaders
        train_dataset = CAMImageDataset(train_x, train_y, train_paths, transform=self.train_transformations[view])
        test_dataset = CAMImageDataset(test_x, test_y, test_paths, transform=self.transformations[view])

        training_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        testing_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

        self.train(num_epochs, training_loader, testing_loader, view, lrate=lrate)

    def cam_loss(self, cam_heatmap, mask):

        """
        Calculates KL divergence loss between CAM heatmaps and binary attention masks.

        Args:
            cam_heatmap (torch.Tensor): Normalized CAM heatmaps of shape [B, H, W].
            mask (torch.Tensor): Binary attention masks of shape [B, H, W].

        Returns:
            torch.Tensor: KL divergence loss.
        """
        # Normalize CAM to make it a probability distribution
        cam_heatmap = cam_heatmap / (cam_heatmap.sum(dim=[1, 2], keepdim=True) + 1e-8)

        # Resize mask if shape mismatch
        if mask.shape != cam_heatmap.shape:
            mask = F.interpolate(
                mask.unsqueeze(1), size=cam_heatmap.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
            
        # Normalize the binary mask to make it a distribution
        mask = mask / (mask.sum(dim=[1, 2], keepdim=True) + 1e-8)

        # Clamp CAM for numerical safety
        cam_heatmap = torch.clamp(cam_heatmap, min=1e-8)

        # Compute KL divergence
        loss = F.kl_div(torch.log(cam_heatmap + 1e-8), mask, reduction='batchmean')
        return loss

    def k_fold_resnet(self, num_epochs, view, k_folds=5, batch=32, rotation=5,
                      brightness=0.1, lrate=0.001, erasing=(0.5, (0.02, 0.15))):
        """
        Trains the model, determined by view, using Stratified K-Fold Cross Validation.

        Args:
            num_epochs (int): Number of epochs to train the model.
            view (str): Image view identifier (e.g., 'caud', 'dors', etc.).
            k_folds (int, optional): Number of cross-validation folds. Default is 5.
            batch (int, optional): Batch size for training and evaluation. Default is 32.
            rotation (int, optional): Maximum degrees of random rotation applied to training images. Default is 5.
            brightness (float, optional): Brightness jitter range for data augmentation. Default is 0.1.
            lrate (float, optional): Learning rate for optimizer. Default is 0.001.
            erasing (tuple, optional): Tuple (p, scale) for RandomErasing:
                - p (float): Probability of applying erasing.
                - scale (tuple): Range of erased area (min %, max % of image area). Default is (0.5, (0.02, 0.15)).

        Returns:
            None
        """
        # Get view dataset(images and labels)
        view_df = self.subsets[view]

        images = view_df[self.image_column].values
        classes = view_df[self.class_column].values
        filenames = view_df["Filename"].values
        labels = [self.class_string_dictionary[label] for label in classes]

        # Define transformation for training
        train_transformations = self.create_train_transformations(
            rotation_degree=rotation,
            brightness=brightness,
            contrast=0.1,
            erasing=erasing
        )
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

        all_fold_f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            print(f"\nFold {fold+1}/{k_folds}:")

            train_x = [images[i] for i in train_idx]
            train_y = [labels[i] for i in train_idx]
            train_paths = [filenames[i] for i in train_idx]

            test_x = [images[i] for i in val_idx]
            test_y = [labels[i] for i in val_idx]
            test_paths = [filenames[i] for i in val_idx]

            # Create DataLoaders
            train_dataset = CAMImageDataset(train_x, train_y, train_paths, transform=train_transformations[view])
            test_dataset = CAMImageDataset(test_x, test_y, test_paths, transform=self.transformations[view])
            train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

            # Reinitialize model before each fold
            self.model_accuracies[view] = 0.0
            model = self.load_model()
            self.models[view] = model

            self.train(num_epochs, train_loader, test_loader, view, lrate=lrate)

            fold_f1 = self.model_accuracies.get(view, 0.0)
            all_fold_f1s.append(fold_f1)
            print(f"Fold {fold+1} Macro F1: {100 * fold_f1:.2f}%")

            # garbage collection and CUDA cache clearing
            torch.cuda.empty_cache()
            gc.collect()

        average_macro_f1 = 100 * sum(all_fold_f1s)/k_folds
        print(f"\nAverage Macro F1 over {k_folds} folds: {average_macro_f1:.2f}%")

    def hyperparameter_training_evaluation(self, num_epochs, train_loader, test_loader, view,
                                           lr, optimizer_type, lambda_attn=None, target_layer="layer4"):
        """
        Trains and evaluates the model for a given view using CAM-guided loss
        and specified hyperparameters.

        Args:
            num_epochs (int): Number of training epochs.
            train_loader (DataLoader): Training data loader.
            test_loader (DataLoader): Evaluation data loader.
            view (str): Image view identifier ('caud', 'dors', etc.)
            lr (float): Learning rate for optimizer.
            optimizer_type (str): Either 'adam' or 'sgd'.
            lambda_attn (float, optional): Weight of CAM-guided attention loss. If None, uses self.lambda_attn.
            target_layer (str): Target layer for Grad-CAM (default: "layer4").

        Returns:
            float: Macro F1 score on the test set.
        """
        model = self.models[view]
        criterion = torch.nn.CrossEntropyLoss()
        lambda_attn = self.lambda_attn if lambda_attn is None else lambda_attn

        # Choose optimizer
        if optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        grad_cam = GradCAM(model, target_layer_module=getattr(model, target_layer))

        for _ in range(num_epochs):
            model.train()
            for inputs, labels, paths in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = model(inputs)
                pred_loss = criterion(outputs, labels)

                # Grad-CAM heatmaps
                cam_heatmaps = grad_cam.generate_heatmap(inputs)
                masks = self.load_attention_masks(paths)

                if masks.sum() == 0:
                    total_loss = pred_loss
                else:
                    attn_loss = self.cam_loss(cam_heatmaps, masks)
                    total_loss = pred_loss + lambda_attn * attn_loss

                total_loss.backward()
                optimizer.step()

        # Evaluate at end of training
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        macro_f1 = f1_score(true_labels, predictions, average="macro")
        return macro_f1

    def cam_optuna_objective(self, trial, view, num_epochs=10, n_splits=3):
        """
        Optuna objective function for CAM training with k-fold cross-validation.

        Args:
            trial: Optuna trial object
            view (str): Image view key ('caud', 'dors', etc.)
            num_epochs (int): Epochs per fold
            n_splits (int): Number of Stratified K-fold splits

        Returns:
            float: Average macro F1 score across folds
        """
        # Sample hyperparameters
        lrate = trial.suggest_loguniform('lrate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        optimizer_type = trial.suggest_categorical('optimizer_type', ['adam', 'sgd'])
        rotation_degree = trial.suggest_int('rotation_degree', 0, 15)
        brightness = trial.suggest_float('brightness', 0.0, 0.5)
        erasing_p = trial.suggest_float('erasing_p', 0.0, 0.8)
        erasing_scale_min = trial.suggest_float('erasing_scale_min', 0.01, 0.1)
        erasing_scale_max = trial.suggest_float('erasing_scale_max', 0.1, 0.4)

        if erasing_scale_min >= erasing_scale_max:
            return 0.0  # Invalid trial

        df_subset = self.subsets[view]
        X = df_subset[self.image_column].values
        filenames = df_subset["Filename"].values
        y = [self.class_string_dictionary[label] for label in df_subset[self.class_column].values]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []

        for train_index, val_index in skf.split(X, y):
            train_x, val_x = X[train_index], X[val_index]
            train_f, val_f = filenames[train_index], filenames[val_index]
            train_y = [y[i] for i in train_index]
            val_y = [y[i] for i in val_index]

            # Update train transformations with sampled params
            self.train_transformations = self.create_train_transformations(
                rotation_degree=rotation_degree,
                brightness=brightness,
                contrast=0.1,
                erasing=(erasing_p, (erasing_scale_min, erasing_scale_max))
            )

            train_dataset = CAMImageDataset(train_x, train_y, train_f, transform=self.train_transformations[view])
            val_dataset = CAMImageDataset(val_x, val_y, val_f, transform=self.transformations[view])

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Reset model weights before training each fold
            self.models[view] = self.load_model()

            # Train & evaluate using CAM-aware tuning method
            macro_f1 = self.hyperparameter_training_evaluation(
                num_epochs=num_epochs,
                train_loader=train_loader,
                test_loader=val_loader,
                view=view,
                lr=lrate,
                optimizer_type=optimizer_type
            )
            fold_scores.append(macro_f1)

            # Free memory after fold
            torch.cuda.empty_cache()
            gc.collect()

        avg_score = sum(fold_scores) / len(fold_scores)
        return avg_score

    def run_cam_optuna_study(self, view, num_trials=20, num_epochs=10):
        """
        Runs hyperparameter tuning with Optuna for a specific image view.

        Args:
            view (str): View key ('caud', 'dors', etc.)
            num_trials (int): Number of trials
            num_epochs (int): Epochs per fold training
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.cam_optuna_objective(trial, view, num_epochs=num_epochs), n_trials=num_trials)

        print(f"Best hyperparameters for {view}: {study.best_params}")
        print(f"Best average Macro F1 for {view}: {100 * study.best_value:.4f}")
        print("Best hyperparameters:", study.best_params)
        return study.best_params

    def load_model(self):
        """
        Loads resnet50 model to be trained and saved
        Return: ResNet model
        """
        model = models.resnet50()
        num_features = model.fc.in_features
        # number of classifications tentative
        model.fc = torch.nn.Linear(num_features, self.num_classes)
        model = model.to(self.device)

        return model

    def save_models(self, model_filenames = None, height_filename = None,
                    class_dict_filename = None, accuracy_dict_filename = None,
                    overwrite_accuracies = False):
        """
        Saves trained models to their respective files and image height file
        
        Returns: None
        """
        # Update/Initialize Model Accuracy Dictionary
        # update_flags indicates which models weights need to be updated and saved
        update_flags = self.update_accuracies(accuracy_dict_filename, overwrite_accuracies)
        views = ["caud", "dors", "fron", "late"]

        for view in views:
            if view in model_filenames and model_filenames[view] and update_flags[view]:
                file = model_filenames[view]
                torch.save(self.models[view].state_dict(), file)
                self.save_transformation(self.transformations[view], view)
                print(f"{view} model weights saved to {file}")

        # Handle dict_filename similarly if needed
        if class_dict_filename:
            with open(class_dict_filename, "w") as file:
                json.dump(self.class_index_dictionary, file, indent=4)
            print(f"Dictionary saved to {class_dict_filename}")

        if height_filename:
            with open(height_filename, "w") as file:
                file.write(str(self.height))
            print(f"Height saved to {height_filename}.")

    def update_accuracies(self, accuracy_dict_filename = None, overwrite_accuracy = False):
        """
        Reads in the previously saved model accuracies(if exists), and updates and saves 
        accuracy dictionary if accuracies increased during training. If model accuracies
        dictionary does not exist, then it initializes with training values or 0 if that
        model was not trained.
        
        Returns: update_flags - dictionary that tracks which models should update their weights
        """

        model_names = ["caud", "dors", "fron", "late"]
        update_flags = {}
        try:
            with open(accuracy_dict_filename, 'r') as f:
                accuracy_dict = json.load(f)

            for model in model_names:
                accuracy = accuracy_dict.get(model, 0)
                # in case of overwrite, make sure that models that weren't trained are not updated
                if self.model_accuracies[model] == 0:
                    update_flags[model] = False
                elif accuracy < self.model_accuracies[model] or overwrite_accuracy:
                    # accuracy from most recent train is better than saved, so update
                    update_flags[model] = True
                    print(f"Updated Accuracy in Dictionary - Improved for {model} model.")
                elif accuracy >= self.model_accuracies[model]:
                    # accuracy did not improve from previously saved accuracy
                    self.model_accuracies[model] = accuracy
                    update_flags[model] = False
                    print(f"No Improvement to Accuracy for {model} model.")

        except FileNotFoundError:
            for model in model_names:
                update_flags[model] = True
            print(f"Accuracy File Not Found - Initializing at {accuracy_dict_filename}")

        with open(accuracy_dict_filename, "w") as file:
            json.dump(self.model_accuracies, file, indent=4)
        print(f"Model accuracies saved to {accuracy_dict_filename}.")

        return update_flags

    def save_transformation(self, transformation, angle):
        """
        Takes a transformation and angle input and saves the transformation
        to a file for use in the evaluation method program

        Returns: None
        """
        with open(f"src/models/cam_{angle}_transformation.pth", "wb") as f:
            dill.dump(transformation, f)

class CAMImageDataset(Dataset):
    """
    Dataset class for CAM-guided training. Returns image, label, and file path.
    """
    def __init__(self, image_binaries, labels, file_paths, transform=None):
        """
        Initializes the dataset.

        Args:
            image_binaries (list[bytes]): Raw image byte data.
            labels (list[int]): Corresponding labels.
            file_paths (list[str]): Paths used to find masks.
            transform (callable, optional): Image transformation pipeline.
        """
        self.image_binaries = image_binaries
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.paths = file_paths
        self.transform = transform

    def __len__(self):
        """Returns the number of samples."""
        return len(self.image_binaries)

    def __getitem__(self, idx):
        """
        Loads and returns a sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label, file_path)
        """
        image_binary = self.image_binaries[idx]
        image = Image.open(BytesIO(image_binary))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.paths[idx]
