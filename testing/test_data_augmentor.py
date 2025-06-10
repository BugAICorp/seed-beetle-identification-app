""" test_data_augmentor """

import unittest
import sys
import os
import pandas as pd
from PIL import Image
from io import BytesIO
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_augmentor import DataAugmenter


class TestDataAugmentor(unittest.TestCase):
    """
    Test the DataAugmentor class methods
    """
    @staticmethod
    def create_dummy_image(color=(255, 0, 0), size=(100, 100)):
        img = Image.new('RGB', size, color)
        with BytesIO() as output:
            img.save(output, format='PNG')
            return output.getvalue()

    @staticmethod
    def dummy_augmentation(pil_img):
        return pil_img.rotate(90)

    def setUp(self):
        blobs = [self.create_dummy_image() for _ in range(5)]
        self.df = pd.DataFrame({
            'Species': ['rare1', 'rare1', 'common', 'common', 'common'],
            'Image': blobs,
            'UniqueID': ['id1', 'id2', 'id3', 'id4', 'id5']
        })

    def test_initialization(self):
        """ Test the initializer to ensure proper setup. """
        augmenter = DataAugmenter(self.df, class_column='Species')
        self.assertTrue(augmenter.df.equals(self.df))
        self.assertEqual(augmenter.class_column, 'Species')
        self.assertEqual(augmenter.image_column, 'Image')
        self.assertEqual(augmenter.id_column, 'UniqueID')
        self.assertEqual(augmenter.threshold, 20)

    def test_get_rare_classes(self):
        """ Test the get_rare_classes method returns the expected classes. """
        augmenter = DataAugmenter(self.df, class_column='Species', threshold=3)
        rare_classes = augmenter.get_rare_classes()
        self.assertIn('rare1', rare_classes)
        self.assertNotIn('common', rare_classes)

    def test_binary_to_pil_and_back(self):
        """ Test that the binary_to_pil and pil_to_binary methods preform as expected. """
        augmenter = DataAugmenter(self.df, class_column='Species')
        original_blob = self.df.iloc[0]['Image']
        pil_img = augmenter.binary_to_pil(original_blob)
        self.assertIsInstance(pil_img, Image.Image)

        new_blob = augmenter.pil_to_binary(pil_img)
        # Assert that it is a binary object, and that it isn't empty
        self.assertIsInstance(new_blob, bytes)
        self.assertNotEqual(new_blob, b'')

    def test_augment_rare_classes_adds_images(self):
        """ Test the augment_rare_classes method correctly adds images to the dataframe. """
        augmenter = DataAugmenter(self.df, class_column='Species', threshold=3)
        augmented_df = augmenter.augment_rare_classes(self.dummy_augmentation, num_augments_per_image=2)

        # Assert that the dataframe is the expected length(2 rare images with 2 augments per image, so 4 additional)
        self.assertEqual(len(augmented_df), len(self.df) + 4)

        new_ids = augmented_df['UniqueID'].tolist()
        self.assertIn('id1_aug0', new_ids)
        self.assertIn('id2_aug1', new_ids)

    def test_no_augmentation_for_common_classes(self):
        augmenter = DataAugmenter(self.df, class_column='Species', threshold=2)
        augmented_df = augmenter.augment_rare_classes(self.dummy_augmentation, num_augments_per_image=1)
        # No augmentations should have been preformed, so size doesn't change
        self.assertEqual(len(augmented_df), len(self.df))


if __name__ == '__main__':
    unittest.main()
