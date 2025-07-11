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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformation_classes import HistogramEqualization

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class GradCAM:
    """
    Implements Grad-CAM for a given model and target layer.
    Used to generate class activation heatmaps for model interpretation.
    """
    def __init__(self, model, target_layer):
        """ 
        Initialize Grad-CAM with model and target layer. 
        
        Args:
            model(torch.nn.Module): The neural network model (ResNet50)
            target_layer(str):The name of the layer inside the model from
                which Grad-CAM will compute the activations.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """ Registers forward and backward hooks to capture activations and gradients. """
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))

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
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        one_hot = torch.zeros_like(output)
        for i, idx in enumerate(class_idx):
            one_hot[i, idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

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

        self.mask_dir = mask_dir
        self.lambda_attn = 0.1
    
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
            mask_img = Image.open(mask_path).convert("L")  # Load as grayscale
            mask_tensor = transforms.ToTensor()(mask_img)  # [1, H, W], range [0.0, 1.0]
            binary_mask = (mask_tensor > 0.5).float()      # Binarize: 1.0 = beetle, 0.0 = background
            masks.append(binary_mask)
        
        # Shape: [B, 1, H, W] → squeeze channel to [B, H, W]
        masks = torch.stack(masks).squeeze(1)
        return masks.to(self.device)
    
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
        grad_cam = GradCAM(model, target_layer=model.layer4)

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
        else:
            return None

    def train_model(self, num_epochs, view, batch, rotation=5, brightness=0.1, lrate=0.001):
        """
        Trains resnet model with subset of specified image views
        and save model to respective save file.
        Return: None
        """
        # Get training and testing data
        train_x, test_x, train_y, test_y = self.get_train_test_split(self.subsets[view])
        # Define image training transformations, placeholder for preprocessing
        self.train_transformations = self.create_train_transformations(
            rotation_degree=rotation,
            brightness=brightness,
            contrast=0.1,
            erasing=(0.5, (0.02, 0.15))
        )

        # Create DataLoaders
        train_dataset = CAMImageDataset(train_x, train_y, train_x, transform=self.train_transformations[view])
        test_dataset = CAMImageDataset(test_x, test_y, test_x, transform=self.transformations[view])

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
        cam_heatmap = cam_heatmap / (cam_heatmap.sum(dim=[1, 2], keepdim=True) + 1e-8)
        if mask.shape != cam_heatmap.shape:
            mask = F.interpolate(mask.unsqueeze(1), size=cam_heatmap.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        loss = F.kl_div(torch.log(cam_heatmap + 1e-8), mask, reduction='batchmean')
        return loss

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
        with open(f"src/models/{angle}_transformation.pth", "wb") as f:
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
