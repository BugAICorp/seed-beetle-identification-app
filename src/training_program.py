""" training_program.py """
import os
import sys
import json
import copy
from io import BytesIO
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch
import dill
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformation_classes import HistogramEqualization
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, unspecified-encoding too-many-public-methods
class TrainingProgram:
    """
    Reads 4 subsets of pandas database from DatabaseReader, and trains and saves 4 models
    according to their respective image angles.
    """
    def __init__(self, dataframe, class_column, num_classes, image_column='Image'):
        """
        Initialize dataset, image height, and individual model training
        Args:
            dataframe (pd.DataFrame): Original dataset with image blobs
            class_column (str): Column header used to determine class
            num_classes (int): Number of classes/outputs for the models
            image_column (str): Column header used to determine the image column
        """
        self.dataframe = dataframe
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
            train_transformations[key] = transforms.Compose([
                # Add augmentations here for testing
                transforms.RandomRotation(degrees=rotation_degree),
                transforms.ColorJitter(brightness=brightness, contrast=contrast),
                transforms.RandomErasing(p=p, scale=scale),
                *self.transformations[key].transforms
            ])

        return train_transformations

    def get_train_test_split(self, df):
        """
        Gets train and test split for given dataframe
        Returns: List of train and test data
        """
        image_binaries = df[self.image_column].values
        classes = df[self.class_column].values
        labels = [self.class_string_dictionary[label] for label in classes]
        # Split subset into training and testing sets
        # x: images, y: species
        train_x, test_x, train_y, test_y = train_test_split(
        image_binaries, labels, test_size=0.2, random_state=42)
        return [train_x, test_x, train_y, test_y]

    def training_evaluation_resnet(self, num_epochs, train_loader, test_loader, view):
        """
        Code for training algorithm and evaluating model
        """
        # Model Training
        # define loss function, optimization function, and image transformation
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.models[view].parameters(), lr=0.001)

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

        # Set model to the best model after training
        if best_state_dict is not None:
            self.models[view].load_state_dict(best_state_dict)
            self.model_accuracies[view] = best_macro_f1
            print(f"Best Macro F1: {100 * best_macro_f1:.2f}% — model loaded.")

    def train_resnet_model(self, num_epochs, view):
        """
        Trains resnet model with subset of specified image views
        and save model to respective save file.
        Return: None
        """
        # Get training and testing data
        train_x, test_x, train_y, test_y = self.get_train_test_split(self.subsets[view])
        # Define image training transformations, placeholder for preprocessing
        transformation = self.train_transformations[view]

        # Create DataLoaders
        train_dataset = ImageDataset(train_x, train_y, transform=transformation)
        test_dataset = ImageDataset(test_x, test_y, transform=transformation)
        training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        testing_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.training_evaluation_resnet(num_epochs, training_loader, testing_loader, view)

    def k_fold_resnet(self, num_epochs, view, k_folds=5):
        """
        Trains the caudal model using Stratified K-Fold Cross Validation.
        """
        # Get view dataset(images and labels)
        view_df = self.subsets[view]

        images = view_df[self.image_column].values
        classes = view_df[self.class_column].values
        labels = [self.class_string_dictionary[label] for label in classes]

        # Define transformation for training
        transformation = self.train_transformations[view]
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        all_fold_f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            print(f"\nFold {fold+1}/{k_folds}:")

            train_x = [images[i] for i in train_idx]
            train_y = [labels[i] for i in train_idx]
            val_x = [images[i] for i in val_idx]
            val_y = [labels[i] for i in val_idx]

            train_dataset = ImageDataset(train_x, train_y, transform=transformation)
            val_dataset = ImageDataset(val_x, val_y, transform=transformation)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Reinitialize model before each fold
            self.model_accuracies[view] = 0.0
            self.models[view] = self.load_model()

            self.training_evaluation_resnet(num_epochs, train_loader, val_loader, view)

            fold_f1 = self.model_accuracies.get(view, 0.0)
            all_fold_f1s.append(fold_f1)

        average_macro_f1 = 100 * sum(all_fold_f1s)/k_folds
        print(f"\nAverage Macro F1 over {k_folds} folds: {average_macro_f1:.2f}%")

    def hyperparameter_training_evaluation(self, num_epochs, train_loader, test_loader, view, lr, optimizer_type):
        """
        Code for training algorithm and evaluating model, adjusted for hyperparameter tuning.
        """
        model = self.models[view]
        criterion = torch.nn.CrossEntropyLoss()

        # Determine optimizer to be used
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Run training algorithm
        for epoch in range(num_epochs):
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
                    class_dict_filename = None, accuracy_dict_filename = None):
        """
        Saves trained models to their respective files and image height file
        
        Returns: None
        """
        # Update/Initialize Model Accuracy Dictionary
        # update_flags indicates which models weights need to be updated and saved
        update_flags = self.update_accuracies(accuracy_dict_filename)
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

    def update_accuracies(self, accuracy_dict_filename = None):
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
                if accuracy < self.model_accuracies[model]:
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
        label (str): species label of image
        transform (transforms.Compose): transform of image to be able 
        to input into model
    """
    def __init__(self, image_binaries, label, transform=None):
        """
        Initialize values
        """
        self.image_binaries = image_binaries
        self.label = torch.tensor(label, dtype=torch.long)
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

        return image, self.label[idx]
