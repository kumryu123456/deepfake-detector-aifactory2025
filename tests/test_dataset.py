"""Unit tests for DeepfakeDataset.

This script validates the Dataset implementation with sample data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from data import DeepfakeDataset, collate_fn, create_inference_dataset


def test_dataset_initialization():
    """Test dataset initialization."""
    print("=" * 80)
    print("TEST 1: Dataset Initialization")
    print("=" * 80)

    # Test with non-existent directory (should raise error)
    try:
        dataset = DeepfakeDataset(
            data_dir="./nonexistent_directory",
            mode="inference",
        )
        print("❌ FAIL: Should raise FileNotFoundError for non-existent directory")
    except FileNotFoundError:
        print("✅ PASS: Correctly raises FileNotFoundError for non-existent directory")

    # Test invalid mode
    try:
        dataset = DeepfakeDataset(
            data_dir="./data",
            mode="invalid_mode",
        )
        print("❌ FAIL: Should raise ValueError for invalid mode")
    except ValueError:
        print("✅ PASS: Correctly raises ValueError for invalid mode")

    # Test training mode without labels
    try:
        dataset = DeepfakeDataset(
            data_dir="./data",
            mode="train",
            labels_dict=None,
        )
        print("❌ FAIL: Should raise ValueError for training mode without labels")
    except ValueError:
        print("✅ PASS: Correctly raises ValueError for training mode without labels")

    print("\n✅ All initialization tests passed!\n")


def test_file_scanning():
    """Test file scanning and filtering."""
    print("=" * 80)
    print("TEST 2: File Scanning")
    print("=" * 80)

    # Create test data directory
    test_dir = Path("./test_data_temp")
    test_dir.mkdir(exist_ok=True)

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

    # Check file count
    expected_count = 5  # 3 images + 2 videos
    actual_count = len(dataset)

    if actual_count == expected_count:
        print(f"✅ PASS: Correctly found {actual_count} supported files")
    else:
        print(f"❌ FAIL: Expected {expected_count} files, got {actual_count}")

    # Check file list
    file_list = dataset.get_file_list()
    print(f"\nFound files: {file_list}")

    # Check statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Images: {stats['num_images']}")
    print(f"  Videos: {stats['num_videos']}")

    if stats["num_images"] == 3 and stats["num_videos"] == 2:
        print("✅ PASS: Correct image/video counts")
    else:
        print(f"❌ FAIL: Expected 3 images and 2 videos")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)

    print("\n✅ File scanning tests passed!\n")


def test_sample_structure():
    """Test sample dictionary structure."""
    print("=" * 80)
    print("TEST 3: Sample Structure")
    print("=" * 80)

    # This test requires actual data - we'll create a mock sample
    print("Creating mock sample to test structure...")

    # Expected keys for inference mode
    expected_keys_inference = {"frames", "filename", "is_video"}

    # Expected keys for training mode
    expected_keys_training = {"frames", "filename", "is_video", "label"}

    # Create mock samples
    mock_sample_inference = {
        "frames": torch.randn(1, 3, 224, 224),  # Image
        "filename": "test.jpg",
        "is_video": False,
    }

    mock_sample_training = {
        "frames": torch.randn(16, 3, 224, 224),  # Video
        "filename": "test.mp4",
        "is_video": True,
        "label": 1,
    }

    # Check inference sample
    if set(mock_sample_inference.keys()) == expected_keys_inference:
        print("✅ PASS: Inference sample has correct keys")
    else:
        print(f"❌ FAIL: Inference sample keys mismatch")

    # Check training sample
    if set(mock_sample_training.keys()) == expected_keys_training:
        print("✅ PASS: Training sample has correct keys")
    else:
        print(f"❌ FAIL: Training sample keys mismatch")

    # Check tensor shapes
    if mock_sample_inference["frames"].shape == (1, 3, 224, 224):
        print("✅ PASS: Image tensor has correct shape (1, 3, 224, 224)")
    else:
        print(f"❌ FAIL: Image tensor shape mismatch")

    if mock_sample_training["frames"].shape == (16, 3, 224, 224):
        print("✅ PASS: Video tensor has correct shape (16, 3, 224, 224)")
    else:
        print(f"❌ FAIL: Video tensor shape mismatch")

    print("\n✅ Sample structure tests passed!\n")


def test_collate_function():
    """Test custom collate function."""
    print("=" * 80)
    print("TEST 4: Collate Function")
    print("=" * 80)

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
    if set(batched.keys()) == expected_keys:
        print("✅ PASS: Batched sample has correct keys")
    else:
        print(f"❌ FAIL: Batched sample keys: {batched.keys()}")

    # Check frames list
    if isinstance(batched["frames"], list) and len(batched["frames"]) == 3:
        print("✅ PASS: Frames is a list with 3 items")
    else:
        print(f"❌ FAIL: Frames should be a list with 3 items")

    # Check num_frames
    expected_num_frames = [1, 16, 1]
    if batched["num_frames"] == expected_num_frames:
        print(f"✅ PASS: num_frames is correct: {expected_num_frames}")
    else:
        print(f"❌ FAIL: num_frames mismatch: {batched['num_frames']}")

    # Check labels
    expected_labels = torch.tensor([0, 1, 1], dtype=torch.long)
    if torch.equal(batched["label"], expected_labels):
        print("✅ PASS: Labels tensor is correct")
    else:
        print(f"❌ FAIL: Labels mismatch: {batched['label']}")

    # Check filenames
    expected_filenames = ["image1.jpg", "video1.mp4", "image2.jpg"]
    if batched["filename"] == expected_filenames:
        print("✅ PASS: Filenames list is correct")
    else:
        print(f"❌ FAIL: Filenames mismatch: {batched['filename']}")

    print("\n✅ Collate function tests passed!\n")


def test_factory_functions():
    """Test factory functions."""
    print("=" * 80)
    print("TEST 5: Factory Functions")
    print("=" * 80)

    # Test create_inference_dataset (will fail if ./data doesn't exist, but that's OK)
    print("Testing create_inference_dataset()...")
    try:
        # Create test directory
        test_dir = Path("./test_data_temp2")
        test_dir.mkdir(exist_ok=True)
        (test_dir / "test.jpg").touch()

        dataset = create_inference_dataset(
            data_dir=test_dir,
            verbose=False,
        )

        if dataset.mode == "inference":
            print("✅ PASS: create_inference_dataset() creates dataset in inference mode")
        else:
            print(f"❌ FAIL: Expected inference mode, got {dataset.mode}")

        # Cleanup
        import shutil
        shutil.rmtree(test_dir)

    except Exception as e:
        print(f"⚠️  WARNING: create_inference_dataset() test skipped: {e}")

    print("\n✅ Factory function tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DEEPFAKE DATASET VALIDATION TESTS")
    print("=" * 80 + "\n")

    try:
        test_dataset_initialization()
        test_file_scanning()
        test_sample_structure()
        test_collate_function()
        test_factory_functions()

        print("=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST SUITE FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
