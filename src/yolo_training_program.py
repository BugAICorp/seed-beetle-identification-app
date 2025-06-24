""" yolo_training_program.py """

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from ultralytics import YOLO
import torchvision.transforms as T
import numpy as np
from globals import yolo_model

class YOLOTrainer:
    """
    YOLOv8 training class for whole image bounding box detection of a single class.

    Args:
        dataset_path (str): Directory with all images.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        img_size (int): Image resize size.
        device (torch.device): Device for training (auto-selected if None).
    """
    def __init__(self, dataset_path, epochs=10, batch_size=8, img_size=512, device=None):
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_built() else
            "cpu"
        )

        # Setup dataset and dataloader
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
        ])

        self.dataset = ImageDataset(self.dataset_path, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # Initialize YOLOv8n model (pretrained)
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=0.001)


    def train(self):
        """
        Runs training loop for the specified number of epochs on the YOLOv8n model.
        Also evaluates accuracy after each epoch using the evaluate_accuracy method.
        """
        for epoch in range(1, self.epochs + 1):
            self.model.model.train()
            running_loss = 0.0

            for images, targets in self.dataloader:
                images = images.to(self.device)

                # Convert labels to ultralytics expected format:
                # list of dicts with 'boxes' and 'cls' tensors
                targets_ultralytics = []
                for b in range(len(targets)):
                    boxes = targets[b][:, 1:].to(self.device)  # bbox coords
                    cls = targets[b][:, 0].long().to(self.device)  # class id
                    targets_ultralytics.append({'boxes': boxes, 'cls': cls})

                loss_dict = self.model.model(images, targets_ultralytics)
                loss = sum(loss_dict.values())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.dataloader)

            # 🔍 Evaluation after epoch
            acc = self.evaluate_accuracy()

            print(f"Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f} - Accuracy: {acc*100:.2f}%")


    def evaluate_accuracy(self, conf_thresh=0.7, iou_thresh=0.7):
        """
        Evaluates model accuracy as the percentage of images
        with a correct detection (IoU and class match).

        A prediction is considered correct if:
        - The predicted confidence is >= conf_thresh.
        - The predicted class matches the ground truth class.
        - The Intersection over Union (IoU) between the predicted box and the ground truth box
        is >= iou_thresh (default 0.5).

        Threshold tuning:
        - conf_thresh: Controls how confident predictions must be to count. Lower values (e.g. 0.3)
        increase recall; higher values (e.g. 0.7) increase precision.
        - iou_thresh: Controls how much overlap is required for a predicted box to count as correct.
        0.5 is standard for balanced evaluation. Higher values (e.g. 0.7) are stricter.

        Returns:
            float: Accuracy (correct / total) across dataset.
        """
        self.model.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.dataloader:
                images = images.to(self.device)

                results = self.model(images, augment=False)
                predictions = results.pred  # list of tensors

                for i in range(len(images)):
                    pred = predictions[i]
                    target = targets[i][0].cpu().numpy()  # [cls, x1, y1, x2, y2]
                    target_cls = int(target[0])
                    target_box = target[1:]

                    # Filter predictions by confidence
                    if pred is None or len(pred) == 0:
                        continue

                    # Find highest scoring box above threshold
                    pred = pred[pred[:, 4] > conf_thresh]
                    if len(pred) == 0:
                        continue

                    # Find best match by IoU
                    for det in pred:
                        pred_cls = int(det[5])
                        pred_box = det[:4].cpu().numpy()
                        iou = self.compute_iou(pred_box, target_box)

                        if pred_cls == target_cls and iou >= iou_thresh:
                            correct += 1
                            break

                    total += 1

        return correct / total if total > 0 else 0.0


    @staticmethod
    def compute_iou(boxA, boxB):
        """
        Computes Intersection over Union (IoU) between two boxes.

        Box format: [x1, y1, x2, y2]
        - (x1, y1) is the top-left coordinate
        - (x2, y2) is the bottom-right coordinate

        IoU is a metric that quantifies the overlap between two boxes. It is calculated as:
        IoU = Area of Intersection / Area of Union

        Returns:
            float: IoU score between 0 and 1.
            - 0 means no overlap
            - 1 means perfect overlap

        Common usage:
            Used in object detection to determine whether a predicted bounding box
            matches a ground truth box (typically IoU ≥ 0.5 is considered a match).
        """

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


    def save(self, save_path=yolo_model):
        """
        Save trained model weights.
        """
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

class ImageDataset(Dataset):
    """
    Custom dataset assuming the entire image is the object (single class).
    Returns image and bounding box covering the full image.
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Label format: [class_id, x_center, y_center, width, height]
        # Full image bbox normalized coords (class 0)
        label = torch.tensor([[0, 0.5, 0.5, 1.0, 1.0]])

        return image, label
