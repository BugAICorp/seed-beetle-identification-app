""" training_program.py """
import os
import sys
import json
import copy
import gc
import math
from io import BytesIO
import dill
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from transformation_classes import HistogramEqualization
from data_augmenter import DataAugmenter
from resnet_dropout_model import ResNet50Dropout
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, unspecified-encoding too-many-public-methods, disable=too-many-lines
class TrainingProgram:
    """
    Reads 4 subsets of pandas database from DatabaseReader, and trains and saves 4 models
    according to their respective image angles.
    """
    def __init__(self, dataframe, class_column, num_classes,
                 image_column='Image', augment=False, balance_classes=0):
        """
        Initialize dataset, image height, and individual model training
        Args:
            dataframe (pd.DataFrame): Original dataset with image blobs
            class_column (str): Column header used to determine class
            num_classes (int): Number of classes/outputs for the models
            image_column (str): Column header used to determine the image column
            augment (bool): Determines if data is augmented or not
            balance_classes (int): Determines if class balancing will be used during training.
                        0 = no balancing
                        1 = class-weighted loss
                        2 = oversampling only (normal loss)
                        3 = both (oversampling + class-weighted loss)
        """
        self.dataframe = dataframe
        self.height = 300
        self.num_classes = num_classes
        # Dataframe variables
        self.image_column = image_column
        self.class_column = class_column
        self.augment = augment

        self.balance_classes = balance_classes

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
        class_to_idx = {label: idx for idx, label in enumerate(sorted(set(classes)))}
        for class_values in classes:
            if class_to_idx[class_values] not in self.class_set:
                self.class_index_dictionary[class_to_idx[class_values]] = class_values
                self.class_string_dictionary[class_values] = class_to_idx[class_values]
                self.class_set.add(class_to_idx[class_values])

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

        self.train_test_indices = {}

    def get_subset(self, view_type, dataframe):
        """
        Reads database and pulls subset where View column is equal to parameter, view_type
        
        Args: view_type (string): View type column value (e.g., 'CAUD', 'DORS', 'FRON', 'LATE')
       
        Return: pd.DataFrame: Subset of database if column value valid, otherwise empty dataframe
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

    def get_train_test_split(self, df):
        """
        Gets train and test split for given dataframe
        Returns: List of train and test data
        """
        # image_binaries = df[self.image_column].values
        classes = df[self.class_column].values
        labels = [self.class_string_dictionary[label] for label in classes]
        # Split by index for safe DataFrame reconstruction
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42
        )
        # Create train/test DataFrames
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        if self.augment:
            augmenter = DataAugmenter(
                dataframe=train_df,
                class_column="Species",
                threshold=100
            )
            train_df = augmenter.augment_rare_classes(num_augments_per_image=5)

        # Extract final train/test values (x: images, y: labels)
        train_x = train_df[self.image_column].values
        train_y = [self.class_string_dictionary[label] for label in train_df[self.class_column].values]

        test_x = test_df[self.image_column].values
        test_y = [self.class_string_dictionary[label] for label in test_df[self.class_column].values]

        return [train_x, test_x, train_y, test_y]

    def get_loss_function(self, train_y):
        """
        Return the loss function based on balancing strategy.
        Uses class-weighted loss if specified.
        """
        if self.balance_classes in [1, 3]:
            train_y = np.array(train_y)

            # Classes present in this training split
            classes_in_split = np.unique(train_y)

            class_weights = compute_class_weight(
                class_weight="balanced", classes=classes_in_split, y=train_y
            )

            # Expand back to full num_classes size
            full_class_weights = np.zeros(self.num_classes, dtype=np.float32)
            full_class_weights[classes_in_split] = class_weights

            class_weights = torch.tensor(full_class_weights, dtype=torch.float).to(self.device)
            return torch.nn.CrossEntropyLoss(weight=class_weights)

        return torch.nn.CrossEntropyLoss()

    def get_train_loader(self, train_dataset, train_y, batch_size, max_os_ratio: float = 3.0):
        """
        Return DataLoader with optional oversampling.
        Handles missing classes by filling zeros for absent classes.
        """
        if self.balance_classes in [2, 3]:
            train_y = np.array(train_y)

            # Count samples per class for all classes
            class_sample_counts = np.zeros(self.num_classes, dtype=np.int64)
            unique, counts = np.unique(train_y, return_counts=True)
            class_sample_counts[unique] = counts

            # Avoid division by zero for missing classes
            weights = np.zeros_like(class_sample_counts, dtype=np.float32)
            nonzero_mask = class_sample_counts > 0
            weights[nonzero_mask] = 1.0 / class_sample_counts[nonzero_mask]

            # Light oversampling safeguard
            # Normalize so max ratio between classes <= max_os_ratio
            min_w, max_w = weights[nonzero_mask].min(), weights[nonzero_mask].max()
            if max_w / min_w > max_os_ratio:
                scale = (min_w * max_os_ratio) / max_w
                weights = np.clip(weights, a_min=None, a_max=scale * weights.max())

            # Assign weight to each sample in train_y
            sample_weights = [weights[t] for t in train_y]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def training_evaluation_resnet(self, num_epochs, train_loader, test_loader, view, train_y, lrate=0.001):
        """
        Code for training algorithm and evaluating model
        """
        # Model Training
        # define loss function, optimization function, and image transformation
        criterion = self.get_loss_function(train_y)
        optimizer = torch.optim.Adam(self.models[view].parameters(), lr=lrate)

        best_epoch = 0
        best_macro_f1 = 0.0
        best_state_dict = None
        for epoch in range(num_epochs):
            self.models[view].train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Clear previous gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.models[view](inputs)
                loss = criterion(outputs, labels)

                # Backpropagation pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

            # evaluate testing machine
            self.models[view].eval()
            correct = 0
            total = 0
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in test_loader:
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

                # Save model if macro_f1 improves
                if macro_f1 > best_macro_f1:
                    best_epoch = epoch + 1
                    best_macro_f1 = macro_f1
                    best_state_dict = copy.deepcopy(self.models[view].state_dict())
                    print(f"Model accuracy improved after epoch {best_epoch}.")
                else:
                    print(f"No improvement to model, the best epoch is {best_epoch}.")

            # Free unused VRAM after each epoch
            torch.cuda.empty_cache()

        # Set model to the best model after training
        if best_state_dict is not None:
            self.models[view].load_state_dict(best_state_dict)
            self.model_accuracies[view] = best_macro_f1
            print(f"Best Macro F1: {100 * best_macro_f1:.2f}% — model loaded.")

    def train_resnet_model(self, num_epochs, view, batch, rotation=5, brightness=0.1, lrate=0.001,
                           erasure_params=None, max_os_ratio: float = 3.0):
        """
        Trains resnet model with subset of specified image views
        and save model to respective save file.
        Return: None
        """
        # Get training and testing data
        train_x, test_x, train_y, test_y = self.get_train_test_split(self.subsets[view])

        # Store test split for correlation analysis
        self.train_test_indices[view] = {
            "test_x": test_x,
            "test_y": test_y
        }

        if erasure_params is not None:
        # Define image training transformations, placeholder for preprocessing
            self.train_transformations = self.create_train_transformations(
                rotation_degree=rotation,
                brightness=brightness,
                contrast=0.1,
                erasing=(erasure_params["p"], (erasure_params["min"], erasure_params["max"]))
            )
        else:
            self.train_transformations = self.create_train_transformations(
                rotation_degree=rotation,
                brightness=brightness,
                contrast=0.1,
                erasing=(0.4, (0.05, 0.25))
            )

        # Create DataLoaders
        train_dataset = ImageDataset(train_x, train_y, transform=self.train_transformations[view])
        test_dataset = ImageDataset(test_x, test_y, transform=self.transformations[view])
        training_loader = self.get_train_loader(train_dataset, train_y, batch, max_os_ratio=max_os_ratio)
        testing_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

        self.training_evaluation_resnet(num_epochs, training_loader, testing_loader, view, train_y=train_y, lrate=lrate)

    def k_fold_resnet(self, num_epochs, view, k_folds=5, batch=32, rotation=5, brightness=0.1, lrate=0.001,
                      erasure_params=None, max_os_ratio: float = 3.0):
        """
        Trains the model(determined by view) using Stratified K-Fold Cross Validation.
        """
        # Get view dataset(images and labels)
        view_df = self.subsets[view]

        images = view_df[self.image_column].values
        classes = view_df[self.class_column].values
        labels = [self.class_string_dictionary[label] for label in classes]

        if erasure_params is not None:
        # Define image training transformations, placeholder for preprocessing
            self.train_transformations = self.create_train_transformations(
                rotation_degree=rotation,
                brightness=brightness,
                contrast=0.1,
                erasing=(erasure_params["p"], (erasure_params["min"], erasure_params["max"]))
            )
        else:
            self.train_transformations = self.create_train_transformations(
                rotation_degree=rotation,
                brightness=brightness,
                contrast=0.1,
                erasing=(0.4, (0.05, 0.25))
            )
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

        all_fold_f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            print(f"\nFold {fold+1}/{k_folds}:")

            train_x = [images[i] for i in train_idx]
            train_y = [labels[i] for i in train_idx]
            val_x = [images[i] for i in val_idx]
            val_y = [labels[i] for i in val_idx]

            if self.augment:
                # Build train dataframe for augmenter
                train_df = view_df.iloc[train_idx].copy()

                # Use DataAugmenter to augment rare classes
                augmenter = DataAugmenter(
                    dataframe=train_df,
                    class_column="Species",
                    threshold=100  # or some threshold appropriate for rarity
                )
                augmented_df = augmenter.augment_rare_classes(num_augments_per_image=5)

                # Extract augmented train data
                train_x = augmented_df[self.image_column].values
                train_y = [self.class_string_dictionary[label] for label in augmented_df[self.class_column].values]

            train_dataset = ImageDataset(train_x, train_y, transform=self.train_transformations[view])
            val_dataset = ImageDataset(val_x, val_y, transform=self.transformations[view])
            train_loader = self.get_train_loader(train_dataset, np.array(train_y), batch, max_os_ratio=max_os_ratio)
            val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

            # Reinitialize model before each fold
            self.model_accuracies[view] = 0.0
            self.models[view] = self.load_model()

            self.training_evaluation_resnet(num_epochs, train_loader, val_loader, view, train_y=train_y, lrate=lrate)

            fold_f1 = self.model_accuracies.get(view, 0.0)
            all_fold_f1s.append(fold_f1)

        average_macro_f1 = 100 * sum(all_fold_f1s)/k_folds
        print(f"\nAverage Macro F1 over {k_folds} folds: {average_macro_f1:.2f}%")

    def hyperparameter_training_evaluation(
            self, num_epochs, train_loader, test_loader, view, train_y, lr, optimizer_type):
        """
        Code for training algorithm and evaluating model, adjusted for hyperparameter tuning.
        Trains and evaluate the model for a given view using specified hyperparameters.

        Args:
            num_epochs (int): Number of epochs to train the model.
            train_loader (DataLoader): DataLoader providing training batches.
            test_loader (DataLoader): DataLoader providing testing/validation batches.
            view (str): Identifier for the model/view to train and evaluate.
            lr (float): Learning rate for the optimizer.
            optimizer_type (str): Optimizer type to use, either 'adam' or 'sgd'.

        Returns:
            float: Macro F1 score computed on the test set predictions.
        
        Raises:
            ValueError: If `optimizer_type` is not supported.
        """
        model = self.models[view]
        criterion = self.get_loss_function(train_y)

        # Determine optimizer to be used
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Run training algorithm
        for _ in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate model at the end rather than at each epoch due to hyperparameter tuning
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        f1 = f1_score(true_labels, predictions, average="macro")
        return f1

    def objective(self, trial, view, num_epochs=10, k_folds=3):
        """
        Objective function for Optuna hyperparameter tuning.

        Suggests hyperparameters including learning rate, batch size, optimizer type,
        and augmentation parameters. Performs k-fold stratified cross-validation
        to evaluate the average macro F1 score of the model under these hyperparameters.

        Args:
            trial (optuna.trial.Trial): Optuna trial object for suggesting hyperparameters.
            view (str): Identifier for the model/view to tune.
            num_epochs (int, optional): Number of training epochs per fold. Defaults to 10.
            k_folds (int, optional): Number of folds for cross-validation. Defaults to 3.

        Returns:
            float: Average macro F1 score across the k folds.
        """
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        rotation = trial.suggest_int("rotation", 0, 20)
        brightness = trial.suggest_float("brightness", 0.0, 0.3)
        erasing_p = trial.suggest_float('erasing_p', 0.0, 0.8)
        erasing_scale_min = trial.suggest_float('erasing_scale_min', 0.01, 0.1)
        erasing_scale_max = trial.suggest_float('erasing_scale_max', 0.1, 0.4)
        max_os_ratio = trial.suggest_float('max_os_ratio', 1.0, 5.0, step=0.5)

        if erasing_scale_min >= erasing_scale_max:
            return 0.0  # Invalid trial

        self.train_transformations = self.create_train_transformations(
            rotation_degree=rotation,
            brightness=brightness,
            contrast=0.1,
            erasing=(erasing_p, (erasing_scale_min, erasing_scale_max))
        )

        # Get view dataset(images and labels)
        view_df = self.subsets[view]

        images = view_df[self.image_column].values
        classes = view_df[self.class_column].values
        labels = [self.class_string_dictionary[label] for label in classes]

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=21)

        all_f1_scores = []

        for train_idx, val_idx in skf.split(images, labels):
            train_x = [images[i] for i in train_idx]
            train_y = [labels[i] for i in train_idx]
            val_x = [images[i] for i in val_idx]
            val_y = [labels[i] for i in val_idx]

            if self.augment:
                # Build train dataframe for augmenter
                train_df = view_df.iloc[train_idx].copy()

                # Use DataAugmenter to augment rare classes
                augmenter = DataAugmenter(
                    dataframe=train_df,
                    class_column="Species",
                    threshold=100  # or some threshold appropriate for rarity
                )
                augmented_df = augmenter.augment_rare_classes(num_augments_per_image=5)

                # Extract augmented train data
                train_x = augmented_df[self.image_column].values
                train_y = [self.class_string_dictionary[label] for label in augmented_df[self.class_column].values]

            train_dataset = ImageDataset(train_x, train_y, transform=self.train_transformations[view])
            val_dataset = ImageDataset(val_x, val_y, transform=self.transformations[view])
            train_loader = self.get_train_loader(
                train_dataset, np.array(train_y), batch_size, max_os_ratio=max_os_ratio)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            self.models[view] = self.load_model()
            f1 = self.hyperparameter_training_evaluation(
                num_epochs=num_epochs,
                train_loader=train_loader,
                test_loader=val_loader,
                view=view,
                train_y=train_y,
                lr=lr,
                optimizer_type="adam"
            )
            all_f1_scores.append(f1)
            # Clear model and loaders
            del train_loader, val_loader
            del self.models[view]
            # clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

        avg_f1 = np.mean(all_f1_scores)
        return avg_f1

    def run_optuna_study(self, view, n_trials=20):
        """
        Run an Optuna hyperparameter optimization study for a specified view.

        Creates and runs an Optuna study to maximize the macro F1 score by
        tuning hyperparameters for the model associated with the given view.
        Prints and returns the best hyperparameters found.

        Args:
            view (str): Identifier for the model/view to optimize.
            n_trials (int, optional): Number of Optuna trials to run. Defaults to 20.

        Returns:
            dict: Best hyperparameters found by the study.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, view), n_trials=n_trials)

        print(f"Best trial for view {view}:")
        print(f"F1 Score: {100 * study.best_value:.2f}%")
        print("Best hyperparameters:", study.best_params)
        return study.best_params

    def mc_dropout_predict(self, view, inputs, n_samples=30):
        """
        Run Monte Carlo Dropout inference for uncertainty estimation.

        Args:
            view (str):
                The dataset view/model to use (e.g., "late", "dors", "caud", "fron").
            inputs (torch.Tensor):
                Input batch of images with shape (N, C, H, W).
            n_samples (int, optional):
                Number of stochastic forward passes to perform using dropout.

        Returns:
            mean_probs (torch.Tensor):
                The averaged class probability predictions across all MC samples.
                Shape: (N, num_classes).

            normalized_entropy (torch.Tensor):
                Normalized Shannon entropy for each sample, representing uncertainty.
                Values range from 0 (high confidence) to 1 (maximum uncertainty).
                Shape: (N,).
        """
        self.models[view].to(self.device)
        self.enable_dropout(view)  # only dropout layers active
        inputs = inputs.to(self.device)

        probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.models[view](inputs)
                probs.append(F.softmax(outputs, dim=1).unsqueeze(0))  # (1, N, C)

        probs = torch.cat(probs, dim=0)        # (n_samples, N, C)
        mean_probs = probs.mean(dim=0)         # (N, C)

        # Raw entropy
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1)
        # Normalize entropy
        num_classes = mean_probs.size(1)
        max_entropy = math.log(num_classes)
        normalized_entropy = entropy / max_entropy

        return mean_probs, normalized_entropy

    def enable_dropout(self, view):
        """ Function to enable dropout layers during test-time """
        for m in self.models[view].modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def evaluate_uncertainty(self, view, n_samples=30, batch_size=32, threshold=None):
        """
        Evaluate model uncertainty on the test split using Monte Carlo Dropout.

        Args:
            view (str): dataset view (e.g. "late", "caud").
            n_samples (int): number of MC dropout forward passes.
            batch_size (int): batch size for evaluation.
            threshold (float, optional): uncertainty cutoff. If set, only keep predictions below this.

        Returns:
            dict: containing predictions, labels, and uncertainties.
        """
        # Get test indices
        test_x = self.train_test_indices[view]["test_x"]
        test_y = self.train_test_indices[view]["test_y"]

        test_dataset = ImageDataset(test_x, test_y, transform=self.transformations[view])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        all_preds, all_labels, all_confidences, all_uncertainties = [], [], [], []

        self.models[view].eval()  # reset model first
        for images, labels in test_loader:
            mean_probs, uncertainty = self.mc_dropout_predict(view, images, n_samples=n_samples)
            preds = mean_probs.argmax(dim=1)
            confidences = mean_probs.max(dim=1).values

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy())

        results = {
            "all_preds": all_preds,
            "all_labels": all_labels,
            "all_confidences": all_confidences,
            "all_uncertainties": all_uncertainties
        }

        # Optionally filter by threshold
        if threshold is not None:
            kept_preds, kept_labels = [], []
            for pred, label, unc in zip(all_preds, all_labels, all_uncertainties):
                if unc < threshold:
                    kept_preds.append(pred)
                    kept_labels.append(label)
            results["filtered_preds"] = kept_preds
            results["filtered_labels"] = kept_labels

        return results

    def create_f1_scores_bar_plot(
            self, view, model=None, batch_size=32, save_path=None, plot=True, plot_save_path=None):
        """
        Analyzes and optionally visualizes per-class F1 scores using the same test split used during training.

        Args:
            view (str): The view ("caud", "dors", etc.)
            model (torch.nn.Module, optional): If None, uses self.models[view]
            batch_size (int): Batch size for evaluation
            save_path (str): Optional path to save F1 scores as a CSV
            plot (bool): Whether to plot a bar chart of F1 scores
            plot_save_path (str): Optional path to save the plot image (e.g., "f1_scores.png")

        Returns:
            pd.DataFrame: Per-class F1 scores in a DataFrame
        """

        if model is None:
            model = self.models[view]

        if not hasattr(self, "train_test_indices") or view not in self.train_test_indices:
            raise ValueError(
                f"No stored train/test split found for view '{view}'. Make sure to call train_resnet_model() first.")

        test_x = self.train_test_indices[view]["test_x"]
        test_y = self.train_test_indices[view]["test_y"]

        test_dataset = ImageDataset(test_x, test_y, transform=self.transformations[view])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Evaluate model
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(targets.cpu().numpy())

        # Get per-class F1
        f1_per_class = f1_score(
            all_true, all_preds,
            average=None,
            labels=list(range(self.num_classes)),
            zero_division=0
        )

        class_names = [self.class_index_dictionary[i] for i in range(self.num_classes)]
        df = pd.DataFrame([f1_per_class], columns=class_names)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved per-class F1 scores for {view} to {save_path}")

        if plot:
            # Dynamically adjust figure height based on number of species
            fig_height = max(6, 0.4 * len(class_names))  # 0.4 inch per class, minimum height = 6
            plt.figure(figsize=(10, fig_height))
            plt.barh(class_names, f1_per_class, color="skyblue")

            plt.xlabel("F1 Score")
            plt.title(f"Per-Class F1 Scores — {view.upper()} View")
            plt.xlim(0, 1.0)
            plt.tight_layout()

            # Add labels to bars
            for i, v in enumerate(f1_per_class):
                plt.text(
                    v + 0.01,  # to the right of the bar
                    i,
                    f"{v:.2f}",
                    va="center"
                )

            if plot_save_path:
                plt.savefig(plot_save_path, bbox_inches="tight")
                print(f"Saved plot for {view} view to {plot_save_path}")
            else:
                plt.show()
            plt.close()

        return df

    def create_confusion_matrix(
            self, view, model=None, batch_size=32, save_path=None, plot=True, plot_save_path=None, normalize=True):
        """
        Generates and optionally visualizes a confusion matrix for a given view using the same test split
        used during training. This can be used to analyze recall and per-class performance.

        Args:
            view (str): The view ("caud", "dors", etc.)
            model (torch.nn.Module, optional): If None, uses self.models[view]
            batch_size (int): Batch size for evaluation
            save_path (str): Optional path to save confusion matrix as CSV
            plot (bool): Whether to plot a heatmap of the confusion matrix
            plot_save_path (str): Optional path to save the plot image
            normalize (bool): Whether to normalize rows to show recall values (0-1)
        Returns:
            pd.DataFrame: Confusion matrix as a DataFrame
        """

        if model is None:
            model = self.models[view]

        if not hasattr(self, "train_test_indices") or view not in self.train_test_indices:
            raise ValueError(
                f"No stored train/test split found for view '{view}'. Make sure to call train_resnet_model() first."
            )

        test_x = self.train_test_indices[view]["test_x"]
        test_y = self.train_test_indices[view]["test_y"]

        test_dataset = ImageDataset(test_x, test_y, transform=self.transformations[view])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Evaluate model
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(targets.cpu().numpy())

        # Compute confusion matrix
        cm = confusion_matrix(all_true, all_preds, labels=list(range(self.num_classes)))
        class_names = [self.class_index_dictionary[i] for i in range(self.num_classes)]

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Replace NaN for classes with no samples

        # Convert to DataFrame for easier saving/viewing
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

        if save_path:
            df_cm.to_csv(save_path)
            print(f"Saved confusion matrix for {view} to {save_path}")

        # Plot heatmap
        if plot:
            self._plot_confusion_matrix(cm, class_names, view, plot_save_path, normalize)

        return df_cm

    def _plot_confusion_matrix(self, cm, class_names, view, plot_save_path, normalize):
        """ Plot and save confusion matrix heatmap. """
        fig_size = max(8, 0.5 * len(class_names))
        plt.figure(figsize=(fig_size, fig_size))

        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.yticks(range(len(class_names)), class_names)
        plt.tick_params(axis='x', which='major', pad=8)
        plt.tick_params(axis='y', which='major', pad=4)

        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        title_str = f"Confusion Matrix — {view.upper()} View"
        if normalize:
            title_str += " (Recall per class)"
        plt.title(title_str)

        # Add numbers to each cell
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                plt.text(j, i, value, ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, left=0.25)

        if plot_save_path:
            plt.savefig(plot_save_path, bbox_inches="tight")
            print(f"Saved confusion matrix plot for {view} to {plot_save_path}")
        else:
            plt.show()
        plt.close()

    def load_model(self):
        """
        Loads ResNet50 model with dropout for MC Dropout uncertainty to be trained and saved
        Return: ResNet model
        """
        # Load ResNet50 model with dropout layers and pretrained weights (weights=True)
        model = ResNet50Dropout(num_classes=self.num_classes, dropout_p=0.5, weights=True)
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
        with open(f"src/models/{angle}_transformation.pth", "wb") as f:
            dill.dump(transformation, f)

# Custom Dataset class for loading images from binary data
class ImageDataset(Dataset):
    """
    Dataset class structure to hold image, transformation,
    and species label
    Arguments:
        image_binaries (0'b): image file in binary values
        labels (str): species label of image
        transform (transforms.Compose): transform of image to be able 
        to input into model
    """
    def __init__(self, image_binaries, labels, transform=None):
        """
        Initialize values
        """
        self.image_binaries = image_binaries
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Return: length of image binary data
        """
        return len(self.image_binaries)

    def __getitem__(self, idx):
        """
        Return: image and respective label
        """
        image_binary = self.image_binaries[idx]
        image = Image.open(BytesIO(image_binary))

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label
