"""Unit tests for DeepfakeDataset.

This script validates the Dataset implementation with sample data.
Run with: pytest tests/test_dataset.py -v
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from data import DeepfakeDataset, collate_fn, create_inference_dataset


class TestDatasetInitialization:
    """Test dataset initialization and validation."""

    def test_nonexistent_directory_raises_error(self):
        """Test that non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DeepfakeDataset(
                data_dir="./nonexistent_directory",
                mode="inference",
            )

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with tempfile.TemporaryDirectory() as test_dir:
            with pytest.raises(ValueError):
                DeepfakeDataset(
                    data_dir=test_dir,
                    mode="invalid_mode",
                )

    def test_train_mode_without_labels_raises_error(self):
        """Test that training mode without labels raises ValueError."""
        with tempfile.TemporaryDirectory() as test_dir:
            with pytest.raises(ValueError):
                DeepfakeDataset(
                    data_dir=test_dir,
                    mode="train",
                    labels_dict=None,
                )


class TestFileScanning:
    """Test file scanning and filtering functionality."""

    def test_file_discovery_and_filtering(self):
        """Test that dataset correctly discovers and filters supported file types."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir = Path(test_dir)

            # Create dummy files
            (test_dir / "image1.jpg").touch()
            (test_dir / "IMAGE2.JPG").touch()  # Uppercase extension
            (test_dir / "image3.png").touch()
            (test_dir / "video1.mp4").touch()
            (test_dir / "VIDEO2.MP4").touch()  # Uppercase extension
            (test_dir / "readme.txt").touch()  # Unsupported format
            (test_dir / "data.csv").touch()  # Unsupported format

            # Initialize dataset
            dataset = DeepfakeDataset(
                data_dir=test_dir,
                mode="inference",
                verbose=False,
            )

            # Check file count (3 images + 2 videos = 5)
            expected_count = 5
            actual_count = len(dataset)
            assert actual_count == expected_count, f"Expected {expected_count} files, got {actual_count}"

            # Check statistics
            stats = dataset.get_statistics()
            assert stats["total_files"] == 5
            assert stats["num_images"] == 3, f"Expected 3 images, got {stats['num_images']}"
            assert stats["num_videos"] == 2, f"Expected 2 videos, got {stats['num_videos']}"

    def test_file_list_retrieval(self):
        """Test that file list can be retrieved."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir = Path(test_dir)

            # Create test files
            (test_dir / "test_image.jpg").touch()
            (test_dir / "test_video.mp4").touch()

            dataset = DeepfakeDataset(
                data_dir=test_dir,
                mode="inference",
                verbose=False,
            )

            file_list = dataset.get_file_list()
            assert isinstance(file_list, list)
            assert len(file_list) == 2


class TestSampleStructure:
    """Test sample dictionary structure and real dataset output."""

    def test_mock_sample_structure_inference(self):
        """Test expected structure for inference mode samples."""
        # Expected keys for inference mode
        expected_keys = {"frames", "filename", "is_video"}

        # Create mock sample
        mock_sample = {
            "frames": torch.randn(1, 3, 224, 224),  # Image
            "filename": "test.jpg",
            "is_video": False,
        }

        assert set(mock_sample.keys()) == expected_keys
        assert mock_sample["frames"].shape == (1, 3, 224, 224)

    def test_mock_sample_structure_training(self):
        """Test expected structure for training mode samples."""
        # Expected keys for training mode
        expected_keys = {"frames", "filename", "is_video", "label"}

        # Create mock sample
        mock_sample = {
            "frames": torch.randn(16, 3, 224, 224),  # Video
            "filename": "test.mp4",
            "is_video": True,
            "label": 1,
        }

        assert set(mock_sample.keys()) == expected_keys
        assert mock_sample["frames"].shape == (16, 3, 224, 224)

    def test_real_dataset_output_image(self):
        """Test actual dataset output with real image file."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir = Path(test_dir)

            # Create a small test image file (1x1 pixel)
            import numpy as np
            from PIL import Image

            test_image_path = test_dir / "test_image.jpg"
            img = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 128)
            img.save(test_image_path)

            # Create dataset
            dataset = DeepfakeDataset(
                data_dir=test_dir,
                mode="inference",
                verbose=False,
            )

            # Get actual sample from dataset
            sample = dataset[0]

            # Verify structure
            assert "frames" in sample
            assert "filename" in sample
            assert "is_video" in sample
            assert isinstance(sample["frames"], torch.Tensor)
            assert sample["frames"].dtype == torch.float32
            assert sample["is_video"] is False
            # Image should have shape (1, C, H, W) for single frame
            assert sample["frames"].ndim == 4
            assert sample["frames"].shape[0] == 1  # Single frame

    def test_real_dataset_output_with_labels(self):
        """Test actual dataset output in training mode with labels."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir = Path(test_dir)

            # Create a small test image
            import numpy as np
            from PIL import Image

            test_image_path = test_dir / "test_image.jpg"
            img = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 128)
            img.save(test_image_path)

            # Create dataset with labels
            labels_dict = {"test_image.jpg": 1}
            dataset = DeepfakeDataset(
                data_dir=test_dir,
                mode="train",
                labels_dict=labels_dict,
                verbose=False,
            )

            # Get sample
            sample = dataset[0]

            # Verify structure includes label
            assert "label" in sample
            assert sample["label"] == 1


class TestCollateFunction:
    """Test custom collate function for batching."""

    def test_collate_mixed_batch(self):
        """Test collate function with mixed image and video batch."""
        # Create mock batch
        batch = [
            {
                "frames": torch.randn(1, 3, 224, 224),  # Image
                "filename": "image1.jpg",
                "is_video": False,
                "label": 0,
            },
            {
                "frames": torch.randn(16, 3, 224, 224),  # Video
                "filename": "video1.mp4",
                "is_video": True,
                "label": 1,
            },
            {
                "frames": torch.randn(1, 3, 224, 224),  # Image
                "filename": "image2.jpg",
                "is_video": False,
                "label": 1,
            },
        ]

        # Apply collate function
        batched = collate_fn(batch)

        # Check structure
        expected_keys = {"frames", "filename", "is_video", "num_frames", "label"}
        assert set(batched.keys()) == expected_keys

        # Check frames list
        assert isinstance(batched["frames"], list)
        assert len(batched["frames"]) == 3

        # Check num_frames
        assert batched["num_frames"] == [1, 16, 1]

        # Check labels tensor
        expected_labels = torch.tensor([0, 1, 1], dtype=torch.long)
        assert torch.equal(batched["label"], expected_labels)

        # Check filenames list
        assert batched["filename"] == ["image1.jpg", "video1.mp4", "image2.jpg"]

    def test_collate_inference_batch(self):
        """Test collate function with inference mode batch (no labels)."""
        batch = [
            {
                "frames": torch.randn(1, 3, 224, 224),
                "filename": "image1.jpg",
                "is_video": False,
            },
            {
                "frames": torch.randn(8, 3, 224, 224),
                "filename": "video1.mp4",
                "is_video": True,
            },
        ]

        batched = collate_fn(batch)

        # Should not have labels key
        assert "label" not in batched
        assert len(batched["frames"]) == 2
        assert batched["num_frames"] == [1, 8]


class TestFactoryFunctions:
    """Test factory functions for dataset creation."""

    def test_create_inference_dataset(self):
        """Test create_inference_dataset factory function."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir = Path(test_dir)
            (test_dir / "test.jpg").touch()

            dataset = create_inference_dataset(
                data_dir=test_dir,
                verbose=False,
            )

            assert dataset.mode == "inference"
            assert len(dataset) >= 0  # Should not raise error


class TestDataLoader:
    """Test DataLoader integration."""

    def test_dataloader_creation(self):
        """Test that dataset works with PyTorch DataLoader."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir = Path(test_dir)

            # Create test files
            import numpy as np
            from PIL import Image

            for i in range(3):
                img = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 128)
                img.save(test_dir / f"image{i}.jpg")

            dataset = DeepfakeDataset(
                data_dir=test_dir,
                mode="inference",
                verbose=False,
            )

            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=collate_fn,
                shuffle=False,
            )

            # Test iteration
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                assert "frames" in batch
                assert "filename" in batch
                assert isinstance(batch["frames"], list)

            assert batch_count > 0  # Should have at least one batch
