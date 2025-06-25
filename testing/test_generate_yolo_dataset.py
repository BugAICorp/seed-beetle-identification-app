""" test_genrate_yolo_dataset.py """

import unittest
import tempfile
import shutil
import sys
import os
from pathlib import Path
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from generate_yolo_dataset import YoloDatasetBuilder

class TestYoloDatasetBuilder(unittest.TestCase):
    """
    Unit tests for the YoloDatasetBuilder class.
    """
    def setUp(self):
        """ Creates a temporary directory with dummy image files """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = Path(self.temp_dir.name) / "source"
        self.source_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            (self.source_dir / f"img_{i}.jpg").write_bytes(b"fake image")

        self.output_dir = Path(self.temp_dir.name) / "output"
        self.test_builder = YoloDatasetBuilder(
            source_dir=self.source_dir,
            output_dir=self.output_dir,
            train_ratio=0.7
        )

    def tearDown(self):
        """ Clean up temporary directory and any other resources. """
        self.temp_dir.cleanup()
        # Also ensure any leftover output_dir or yaml file is removed
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        yaml_path = Path(self.test_builder.yaml)
        if yaml_path.exists():
            yaml_path.unlink()

    def test_build(self):
        """ Test build method creates correct directory structure and YAML """
        self.test_builder.build(total_images=6)

        # Check train/val directories exist
        self.assertTrue((self.output_dir / "train/images").exists())
        self.assertTrue((self.output_dir / "val/images").exists())
        self.assertTrue((self.output_dir / "train/labels").exists())
        self.assertTrue((self.output_dir / "val/labels").exists())

        # Check images and labels were created
        train_images = list((self.output_dir / "train/images").glob("*.jpg"))
        val_images = list((self.output_dir / "val/images").glob("*.jpg"))
        self.assertEqual(len(train_images) + len(val_images), 6)

        # Check corresponding label files exist
        for img in train_images + val_images:
            label_path = img.parent.parent / "labels" / (img.stem + ".txt")
            self.assertTrue(label_path.exists())
            with open(label_path) as f:
                self.assertEqual(f.read().strip(), "0 0.5 0.5 1.0 1.0")

        # Check YAML file
        self.assertTrue(Path(self.test_builder.yaml).exists())
        with open(self.test_builder.yaml, 'r') as f:
            data = yaml.safe_load(f)
        self.assertIn("train", data)
        self.assertIn("val", data)
        self.assertEqual(data["names"], ["Seed Beetle"])

    def test_cleanup(self):
        """ Test cleanup deletes created directories and YAML file. """
        self.test_builder.build(total_images=4)
        self.test_builder.cleanup()

        self.assertFalse(self.output_dir.exists())
        self.assertFalse(Path(self.test_builder.yaml).exists())

if __name__ == '__main__':
    unittest.main()
