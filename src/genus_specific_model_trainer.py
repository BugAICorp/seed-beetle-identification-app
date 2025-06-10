"""genus_specific_model_trainer.py"""
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformation_classes import HistogramEqualization
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class GenusSpecificModelTrainer:
    """
    Creates a model trained on the species of each genus individually resulting in
    more models, but with less outputs per model
    """
    def __init__(self, dataframe):
        """
        Initialize variables for assisting training
        """
        self.dataframe = dataframe
        self.height = 300

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
    
    def get_train_test_split(self, dataframe, class_string_dictionary):
        """
        Gets train and test split for given dataframe
        Returns: list of train and test data
        """
        image_binaries = dataframe.iloc[:, -1].values
        classes = dataframe.iloc[:, 1].values
        labels = [class_string_dictionary[label] for label in classes]
        train_x, test_x, train_y, test_y = train_test_split(
            image_binaries, labels, test_size=0.2, random_state=42
        )
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
            return None
        
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
        train_dataset = ImageDataset(train_x, train_y, transform=self.transformation)
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

    def save_model(self, model, genus, class_dict):
        """
        Saves trained models and their files
        """
        torch.save(model.state_dict(), f"src/genus_models/{genus}_species.pth")
        print(f"Model saved to {genus}_species.pth")

        with open(f"src/genus_models/{genus}_dict.json", "w") as file:
            json.dump(class_dict, file, indent=4)
        print(f"Dictionary saved to {genus}_dict.json")

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
            print(f"Accuracy file not found")

        with open(accuracy_dict, 'w') as file:
            json.dump(acc_dict, file, indent=4)
        print(f"Accuracy saved to {accuracy_dict}")

        return update


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
