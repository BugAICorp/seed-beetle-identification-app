""" ood_tester.py """

import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from PIL import Image

class ImageDataset(Dataset):
    """
    A dataset class that wraps a pandas DataFrame containing image data.
    Expects a column 'Image' with either PIL Image, numpy array, or raw bytes.
    """
    def __init__(self, dataframe, transform=None):
        """
        Initialize values

        Arguments:
            dataframe (pd.DataFrame): Dataframe with image blobs
            transform (transforms.Compose): transform of image to be able 
            to input into model
        """
        self.df = dataframe
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_data = self.df.iloc[idx]['Image']

        # Convert raw bytes -> PIL
        if isinstance(img_data, bytes):
            img_data = Image.open(io.BytesIO(img_data)).convert('RGB')

        # Convert numpy -> PIL
        if isinstance(img_data, np.ndarray):
            img_data = Image.fromarray(img_data).convert('RGB')

        # Assume already PIL otherwise
        if self.transform:
            img_data = self.transform(img_data)

        return img_data

class OODTester:
    """
    A utility class for evaluating out-of-distribution (OOD) detection
    using energy-based scores on a given model and datasets.
    """

    def __init__(self, model, id_dataframe, ood_dataframe, transform):
        """
        Args:
            model (torch.nn.Module): Trained model.
            id_df (pd.DataFrame): DataFrame for in-distribution data.
            ood_df (pd.DataFrame): DataFrame for out-of-distribution data.
            batch_size (int): Batch size.
            num_workers (int): DataLoader workers.
            transform (callable): Image transform (default: ToTensor).
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_built()
            else 'cpu')
        self.model = model.to(self.device)

        id_dataset = ImageDataset(id_dataframe, transform=transform)
        ood_dataset = ImageDataset(ood_dataframe, transform=transform)

        # shuffe=false for deterministic results
        self.id_loader = DataLoader(id_dataset, batch_size=32, shuffle=False)
        self.ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False)

    def compute_energy(self, logits, temperature):
        """
        Computes the energy score for a batch of logits at a given temperature.

        Args:
            logits (Tensor): Raw model logits of shape (batch_size, num_classes).
            temperature (float): Temperature value for energy scaling.

        Returns:
            Tensor: Energy scores of shape (batch_size,).
        """
        return -temperature * torch.logsumexp(logits / temperature, dim=1)

    def get_energy_scores(self, dataloader, temperature):
        """
        Computes energy scores for all samples in a dataloader.

        Args:
            dataloader (DataLoader): PyTorch DataLoader for ID or OOD data.
            temperature (float): Temperature value for energy computation.

        Returns:
            np.ndarray: Concatenated energy scores for all samples.
        """
        self.model.eval()
        energies = []

        with torch.no_grad():
            for inputs in dataloader:
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]  # remove labels if present
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                energy = self.compute_energy(logits, temperature)
                energies.append(energy.cpu().numpy())

        return np.concatenate(energies)

    def test_ood(self, temperatures):
        """
        Tests OOD detection performance across multiple temperatures.

        Args:
            temperatures (list of float): List of temperatures to evaluate.

        Returns:
            Tuple[float, dict]: 
                - Best temperature (float) based on AUROC.
                - Results dict containing energy scores, AUROC, and AUPR for each temperature.
        """
        best_auroc = -1
        best_temp = None
        results = {}

        for T in temperatures:
            print(f"Testing temperature {T}")
            id_energies = self.get_energy_scores(self.id_loader, T)
            ood_energies = self.get_energy_scores(self.ood_loader, T)

            labels = np.concatenate([np.ones_like(id_energies), np.zeros_like(ood_energies)])
            scores = np.concatenate([id_energies, ood_energies])

            auroc = roc_auc_score(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            aupr = auc(recall, precision)

            results[T] = {
                'id_energies': id_energies,
                'ood_energies': ood_energies,
                'auroc': auroc,
                'aupr': aupr
            }

            print(f"Temperature {T} -> AUROC: {auroc:.3f}, AUPR: {aupr:.3f}")

            if auroc > best_auroc:
                best_auroc = auroc
                best_temp = T

        print(f"\nBest temperature: {best_temp} with AUROC={best_auroc:.3f}")
        return best_temp, results

    def plot_distributions(self, id_energies, ood_energies, temperature):
        """
        Plots histograms of energy score distributions for ID and OOD data.

        Args:
            id_energies (np.ndarray): Energy scores for in-distribution data.
            ood_energies (np.ndarray): Energy scores for out-of-distribution data.
            temperature (float): Temperature value corresponding to the scores.
        """
        plt.hist(id_energies, bins=50, alpha=0.5, label='ID', density=True)
        plt.hist(ood_energies, bins=50, alpha=0.5, label='OOD', density=True)
        plt.xlabel('Energy Score')
        plt.ylabel('Density')
        plt.title(f'Energy Score Distribution (T={temperature})')
        plt.legend()
        plt.show()
