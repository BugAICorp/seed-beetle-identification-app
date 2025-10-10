""" resnet_dropout_model.py """

import torch
from torchvision import models


class ResNet50Dropout(torch.nn.Module):
    """
    ResNet50 model with added dropout layers for Monte Carlo Dropout.
    This class extends the torchvision ResNet50 by inserting dropout layers
    after the third residual block (`layer3`), after the fourth residual block
    (`layer4`), and before the final fully connected classifier.
    Attributes:
        features (torch.nn.Sequential): Convolutional and residual layers
            of ResNet50 with inserted dropout.
        classifier (torch.nn.Sequential): Flatten + dropout + linear
            classification head.
    """
    def __init__(self, num_classes, dropout_p=0.5, weights=True):
        """
        Constructor for ResNet50Dropout class.
        Args:
            num_classes (int): Number of output classes for classification.
            dropout_p (float, optional): Dropout probability (default=0.5).
            weights (bool):
                - True: Use ImageNet pretrained weights
                - False: Initialize with random weights
        """
        super().__init__()
        if weights:
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            base_model = models.resnet50(weights=None)

        # Keep all layers except the final FC
        self.features = torch.nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            torch.nn.Dropout(p=dropout_p),   # Dropout after layer3
            base_model.layer4,
            torch.nn.Dropout(p=dropout_p),   # Dropout after layer4
            base_model.avgpool,
        )

        num_features = base_model.fc.in_features
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(p=dropout_p),   # Dropout before FC
            torch.nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W).
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
