""" data_augmenter.py """

from io import BytesIO
import pandas as pd
from PIL import Image
from torchvision import transforms

class DataAugmenter:
    """
    Takes in a dataframe and augments more images for rare classes using the passed
    class column, image column, and threshold.
    """
    def __init__(self, dataframe, class_column, image_column='Image', id_column='UniqueID', threshold=20):
        """
        Args:
            dataframe (pd.DataFrame): Original dataset with image blobs
            class_column (str): Column index used to determine class
            image_column (str): Column name containing image BLOBs
            id_column (str): Column with unique identifiers (e.g. UUID or filename)
            threshold (int): Max number of instances a class can have before it's considered common
        """
        self.df = dataframe
        self.class_column = class_column
        self.image_column = image_column
        self.id_column = id_column
        self.threshold = threshold
        # Create minor image transformation for newly generated images
        self.transformation = transforms.Compose([
                transforms.RandomRotation(degrees=5)
            ])

    def binary_to_pil(self, binary_blob):
        """
        Converts binary image data to a PIL Image
        """
        return Image.open(BytesIO(binary_blob)).convert("RGB")

    def pil_to_binary(self, image):
        """
        Converts a PIL Image to binary format
        """
        with BytesIO() as output:
            image.save(output, format='PNG')
            return output.getvalue()

    def get_rare_classes(self):
        """
        Returns a list of class labels that appear fewer times than the threshold
        """
        class_counts = self.df[self.class_column].value_counts()
        rare_classes = class_counts[class_counts < self.threshold].index.tolist()
        return rare_classes

    def augment_rare_classes(self, num_augments_per_image=1):
        """
        Augments the dataset for rare classes
        """
        rare_classes = self.get_rare_classes()
        augmented_rows = []

        for _, row in self.df.iterrows():
            if row[self.class_column] in rare_classes:
                original_blob = row[self.image_column]
                original_id = row[self.id_column]
                pil_img = self.binary_to_pil(original_blob)

                for i in range(num_augments_per_image):
                    # Apply augmentation
                    augmented_img = self.transformation(pil_img)
                    augmented_blob = self.pil_to_binary(augmented_img)

                    # Copy row and just replace the Image
                    new_row = row.copy()
                    new_row[self.image_column] = augmented_blob
                    new_row[self.id_column] = f"{original_id}_aug{i}"
                    augmented_rows.append(new_row)

        # Combine augmented data with the original dataframe and return
        augmented_df = pd.DataFrame(augmented_rows)
        return pd.concat([self.df, augmented_df], ignore_index=True)
