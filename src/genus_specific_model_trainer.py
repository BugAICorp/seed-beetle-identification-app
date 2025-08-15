"""genus_specific_model_trainer.py"""
import os
import sys
import json
import copy
import gc
from io import BytesIO
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch
import dill
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import optuna
from transformation_classes import HistogramEqualization
from data_augmenter import DataAugmenter
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class GenusSpecificModelTrainer:
    """
    Creates a model trained on the species of each genus individually resulting in
    more models, but with less outputs per model
    """
    def __init__(self, dataframe, image_column='Image', augment=False):
        """
        Initialize variables for assisting training
        """
        self.dataframe = dataframe
        self.height = 300
        self.image_column = image_column
        self.augment = augment

        self.model_accuracies = {}

        # Set device to a CUDA-compatible gpu, else mps, else cpu
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_built()
            else 'cpu')

        self.transformation = transforms.Compose([
            transforms.Resize((self.height, self.height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def get_subset(self, genus_type, dataframe):
        """
        Read the database and pull the data associated with a specific Genus

        Args: genus_type (string) : genus column value
                dataframe (pd.Dataframe) : dataframe to analyze

        Returns: pd.Dataframe : Subset of database if genus column is valid
        """
        return dataframe[dataframe["Genus"] == genus_type] if not dataframe.empty else pd.DataFrame()

    def load_model(self, num_classes):
        """
        Loads a model to be used in training
        Args: num_classes(int): number of species in the genus' scope
        Returns: ResNet model
        """
        model = models.resnet50()
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
        model = model.to(self.device)

        return model

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

        base_transform = self.transformation.transforms

        # Manually reorder to insert augmentations in the correct locations
        new_transforms = []
        normalize_transform = None
        for t in base_transform:
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

        train_transformation = transforms.Compose(new_transforms)

        return train_transformation

    def get_train_test_split(self, dataframe, class_string_dictionary):
        """
        Gets train and test split for given dataframe
        Returns: list of train and test data
        """
        classes = dataframe.iloc[:, 1].values
        labels = [class_string_dictionary[label] for label in classes]

        # Split by index for safe DataFrame reconstruction
        indices = np.arange(len(dataframe))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42
        )
        # Create train/test DataFrames
        train_df = dataframe.iloc[train_idx].copy()
        test_df = dataframe.iloc[test_idx].copy()

        if self.augment:
            augmenter = DataAugmenter(
                dataframe=train_df,
                class_column="Species",
                threshold=100
            )
            train_df = augmenter.augment_rare_classes(num_augments_per_image=5)

        # Extract final train/test values (x: images, y: labels)
        train_x = train_df[self.image_column].values
        train_y = [class_string_dictionary[label] for label in train_df["Species"].values]

        test_x = test_df[self.image_column].values
        test_y = [class_string_dictionary[label] for label in test_df["Species"].values]

        return [train_x, test_x, train_y, test_y]

    def train_genus(self, genus, num_epochs):
        """
        Handles isolation of and training of species under a specified genus

        Args: genus(string): genus to train
                num_epochs(int): number of epochs of training to run
        
        Returns: None
        """
        #Pull the Genus' subset, count num of species in subset, and prep model/dict
        genus_subset = self.get_subset(genus, self.dataframe)
        num_species = genus_subset['Species'].nunique()
        print(f"Number of species in {genus} in dataset: {num_species}")
        if num_species < 1:
            return

        genus_model = self.load_model(num_species)

        #Set up index tracking for classifications
        class_index_dictionary = {}
        class_string_dictionary = {}
        class_set = set()
        classes = genus_subset.iloc[:, 1].values
        class_to_idx = {label: idx for idx, label in enumerate(sorted(set(classes)))}
        for class_values in classes:
            if class_to_idx[class_values] not in class_set:
                class_index_dictionary[class_to_idx[class_values]] = class_values
                class_string_dictionary[class_values] = class_to_idx[class_values]
                class_set.add(class_to_idx[class_values])

        train_x, test_x, train_y, test_y = self.get_train_test_split(genus_subset, class_string_dictionary)
        # Define image training transformations, placeholder for preprocessing
        self.train_transformations = self.create_train_transformations(
            rotation_degree=5,
            brightness=0.1,
            contrast=0.1,
            erasing=(0.5, (0.02, 0.15))
        )

        train_dataset = ImageDataset(train_x, train_y, transform=self.train_transformations)
        test_dataset = ImageDataset(test_x, test_y, transform=self.transformation)
        training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        testing_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.training_evaluation(num_epochs, training_loader, testing_loader, genus_model, genus)

        update_model = self.update_accuracies(genus, globals.genus_specific_accuracies)

        if update_model:
            self.save_model(genus_model, genus, class_index_dictionary)


    def training_evaluation(self, num_epochs, train_loader, test_loader, model, genus):
        """
        Code for training and evaluating the specified model
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_epoch = 0
        best_macro_f1 = 0.0
        best_state_dict = None
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader): .4f}")

            model.eval()
            correct = 0
            total = 0
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            if total != 0:
                accuracy = correct / total
                print(f"Accuracy: {100 * accuracy:.2f}%")
                weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
                macro_f1 = f1_score(all_labels, all_predictions, average='macro')
                print(f"Weighted F1 Score: {100 * weighted_f1:.2f}%")
                print(f"Macro F1 Score: {100 * macro_f1:.2f}%")

                if macro_f1 > best_macro_f1:
                    best_epoch = epoch + 1
                    best_macro_f1 = macro_f1
                    best_state_dict = copy.deepcopy(model.state_dict())
                    print(f"Model accuracy improved after epoch {best_epoch}")
                else:
                    print(f"No improvement this epoch, best epoch: {best_epoch}")

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            self.model_accuracies[genus] = best_macro_f1
            print(f"Best Macro F1: {100 * best_macro_f1:.2f}% - model loaded.")


    def gs_hyperparameter_training_evaluation(self, num_epochs, train_loader, test_loader, input_model, lr):
        """
        Code for alternate training algorithm and evaluating model, adjusted for hyperparameter tuning.
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
        model = input_model
        criterion = torch.nn.CrossEntropyLoss()

        # Determine optimizer to be used
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    def objective(self, trial, genus, num_epochs=10, k_folds=3):
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

        train_transformation = self.create_train_transformations(
            rotation_degree=rotation,
            brightness=brightness,
            contrast=0.1,
            erasing=(0.5, (0.02, 0.15))
        )

        # Get view dataset(images and labels)
        genus_df = self.get_subset(genus, self.dataframe)

        images = genus_df[self.image_column].values

        #Set up index tracking for classifications
        class_index_dictionary = {}
        class_string_dictionary = {}
        class_set = set()
        classes = genus_df.iloc[:, 1].values
        class_to_idx = {label: idx for idx, label in enumerate(sorted(set(classes)))}
        for class_values in classes:
            if class_to_idx[class_values] not in class_set:
                class_index_dictionary[class_to_idx[class_values]] = class_values
                class_string_dictionary[class_values] = class_to_idx[class_values]
                class_set.add(class_to_idx[class_values])
        labels = [class_string_dictionary[label] for label in classes]

        # Define transformation for training
        transformation = train_transformation

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=21)

        all_f1_scores = []

        for train_idx, val_idx in skf.split(images, labels):
            train_x = [images[i] for i in train_idx]
            train_y = [labels[i] for i in train_idx]
            val_x = [images[i] for i in val_idx]
            val_y = [labels[i] for i in val_idx]

            if self.augment:
                # Build train dataframe for augmenter
                train_df = genus_df.iloc[train_idx].copy()

                # Use DataAugmenter to augment rare classes
                augmenter = DataAugmenter(
                    dataframe=train_df,
                    class_column="Species",
                    threshold=100  # or some threshold appropriate for rarity
                )
                augmented_df = augmenter.augment_rare_classes(num_augments_per_image=5)

                # Extract augmented train data
                train_x = augmented_df[self.image_column].values
                train_y = [class_string_dictionary[label] for label in augmented_df["Species"].values]

            train_dataset = ImageDataset(train_x, train_y, transform=transformation)
            val_dataset = ImageDataset(val_x, val_y, transform=transformation)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            num_species = genus_df['Species'].nunique()
            model = self.load_model(num_species)
            f1 = self.gs_hyperparameter_training_evaluation(
                num_epochs=num_epochs,
                train_loader=train_loader,
                test_loader=val_loader,
                input_model=model,
                lr=lr
            )
            all_f1_scores.append(f1)
            # Clear model and loaders
            del train_loader, val_loader
            del model
            # clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

        avg_f1 = np.mean(all_f1_scores)
        return avg_f1

    def run_optuna_study(self, genus, n_trials=20):
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
        study.optimize(lambda trial: self.objective(trial, genus), n_trials=n_trials)

        print(f"Best trial for species {genus}:")
        print(f"F1 Score: {100 * study.best_value:.2f}%")
        print("Best hyperparameters:", study.best_params)
        return study.best_params

    def save_model(self, model, genus, class_dict):
        """
        Saves trained models and their files
        """
        torch.save(model.state_dict(), f"src/genus_models/{genus}_species.pth")
        print(f"Model saved to {genus}_species.pth")

        with open(f"src/genus_models/{genus}_dict.json", "w") as file:
            json.dump(class_dict, file, indent=4)
        print(f"Dictionary saved to {genus}_dict.json")

        self.save_transformation()

    def update_accuracies(self, genus, accuracy_dict):
        """
        Checks previous saved accuracies and updates and saves an accuracy dictionary
        if they improved via training. If they do not exist, then initialize with
        training values

        Returns: False if model should not be saved, True if it should
        """
        acc_dict = None
        update = False
        try:
            with open(accuracy_dict, 'r') as f:
                acc_dict = json.load(f)

                cur_acc = self.model_accuracies[genus]
                prev_acc = 0
                if genus in acc_dict:
                    prev_acc = acc_dict.get(genus)

                if cur_acc > prev_acc:
                    update = True
                    acc_dict[genus] = cur_acc

        except FileNotFoundError:
            update = True
            acc_dict = {genus: self.model_accuracies[genus]}
            print("Accuracy file not found")

        with open(accuracy_dict, 'w') as file:
            json.dump(acc_dict, file, indent=4)
        print(f"Accuracy saved to {accuracy_dict}")

        return update

    def save_transformation(self):
        """Saves the transformation used for all species models"""
        with open(globals.gs_transformation, "wb") as f:
            dill.dump(self.transformation, f)


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
