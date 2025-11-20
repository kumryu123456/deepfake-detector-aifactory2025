"""Dataset module for deepfake detection.

This module provides PyTorch Dataset classes for loading and preprocessing
images and videos from the competition data directory.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .face_detector import FaceDetector
from .transforms import DataPreprocessor
from .video_processor import VideoProcessor


class DeepfakeDataset(Dataset):
    """PyTorch Dataset for deepfake detection competition.

    Handles mixed image and video files from ./data/ directory.
    Supports both training and inference modes.

    Competition Requirements:
    - Data location: ./data/ (no subdirectories)
    - Supported formats: JPG, PNG, MP4 (case-insensitive)
    - Output: One prediction per file (videos aggregated)

    Example:
        >>> # Inference mode
        >>> dataset = DeepfakeDataset(
        ...     data_dir="./data",
        ...     mode="inference",
        ...     face_detector_name="mtcnn",
        ... )
        >>> sample = dataset[0]
        >>> # For images: sample["frames"] has shape (1, 3, 224, 224)
        >>> # For videos: sample["frames"] has shape (N, 3, 224, 224)

        >>> # Training mode
        >>> dataset = DeepfakeDataset(
        ...     data_dir="./data/train",
        ...     mode="train",
        ...     labels_dict={"fake_001.jpg": 1, "real_002.mp4": 0, ...},
        ... )
    """

    # Supported file extensions (case-insensitive)
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    VIDEO_EXTENSIONS = {".mp4"}

    def __init__(
        self,
        data_dir: Union[str, Path],
        mode: str = "inference",
        labels_dict: Optional[Dict[str, int]] = None,
        face_detector_name: str = "mtcnn",
        face_confidence_threshold: float = 0.9,
        face_fallback_strategy: str = "center_crop",
        video_target_frames: int = 16,
        video_sampling_method: str = "uniform",
        input_size: Tuple[int, int] = (224, 224),
        augmentation_config: Optional[Dict] = None,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """Initialize DeepfakeDataset.

        Args:
            data_dir: Path to data directory containing images and videos
            mode: "train" or "inference"
                - train: Requires labels_dict, applies augmentation
                - inference: No labels, minimal preprocessing
            labels_dict: Dictionary mapping filename → label (0=Real, 1=Fake)
                Required for training mode, ignored for inference
            face_detector_name: Face detector backend ("mtcnn", "retinaface", "mediapipe")
            face_confidence_threshold: Minimum confidence for face detection
            face_fallback_strategy: What to do when no face detected
                - "center_crop": Use center crop as fallback (recommended)
                - "skip": Skip the file entirely
                - "error": Raise error
            video_target_frames: Number of frames to extract per video
            video_sampling_method: Frame sampling strategy ("uniform", "random", "adaptive")
            input_size: Target image size (height, width)
            augmentation_config: Augmentation configuration (for training mode)
            device: Device for face detection ("cuda" or "cpu")
            verbose: Print progress information
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.labels_dict = labels_dict or {}
        self.input_size = input_size
        self.device = device
        self.verbose = verbose

        # Validate mode
        if mode not in ["train", "inference"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'")

        # Validate labels for training mode
        if mode == "train" and not labels_dict:
            raise ValueError("labels_dict is required for training mode")

        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Initialize face detector
        self.face_detector = FaceDetector(
            detector_name=face_detector_name,
            confidence_threshold=face_confidence_threshold,
            margin_ratio=0.3,
            device=device,
            fallback_strategy=face_fallback_strategy,
        )

        # Initialize video processor
        self.video_processor = VideoProcessor(
            target_frames=video_target_frames,
            sampling_method=video_sampling_method,
            face_detector=self.face_detector,
            show_progress=verbose,
        )

        # Initialize data preprocessor
        self.preprocessor = DataPreprocessor(
            mode=mode,
            input_size=input_size,
            augmentation_config=augmentation_config,
        )

        # Scan data directory and build file list
        self.file_list = self._scan_data_directory()

        if verbose:
            print(f"DeepfakeDataset initialized:")
            print(f"  Mode: {mode}")
            print(f"  Data directory: {self.data_dir}")
            print(f"  Total files: {len(self.file_list)}")
            print(f"  Images: {sum(1 for f in self.file_list if self._is_image(f))}")
            print(f"  Videos: {sum(1 for f in self.file_list if self._is_video(f))}")

    def _scan_data_directory(self) -> List[Path]:
        """Scan data directory for supported files.

        Returns:
            List of file paths (sorted for consistency)
        """
        files = []

        # Iterate through all files in data directory (no subdirectories)
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()

                # Check if supported format
                if ext in self.IMAGE_EXTENSIONS or ext in self.VIDEO_EXTENSIONS:
                    files.append(file_path)

        # Sort for consistency
        files = sorted(files, key=lambda x: x.name)

        if len(files) == 0:
            import warnings
            warnings.warn(
                f"No supported files found in {self.data_dir}. "
                f"Supported formats: {self.IMAGE_EXTENSIONS | self.VIDEO_EXTENSIONS}"
            )

        return files

    def _is_image(self, file_path: Path) -> bool:
        """Check if file is an image."""
        return file_path.suffix.lower() in self.IMAGE_EXTENSIONS

    def _is_video(self, file_path: Path) -> bool:
        """Check if file is a video."""
        return file_path.suffix.lower() in self.VIDEO_EXTENSIONS

    def _process_image(self, file_path: Path) -> Optional[torch.Tensor]:
        """Process image file.

        Args:
            file_path: Path to image file

        Returns:
            Preprocessed image tensor of shape (3, H, W), or None if failed
        """
        try:
            # Read image
            image = cv2.imread(str(file_path))
            if image is None:
                if self.verbose:
                    print(f"Warning: Failed to read image: {file_path.name}")
                return None

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect and crop face
            face = self.face_detector.detect_and_crop(image, target_size=self.input_size)

            if face is None:
                if self.verbose:
                    print(f"Warning: No face detected in {file_path.name}")
                return None

            # Preprocess (normalize and convert to tensor)
            tensor = self.preprocessor(face)  # Shape: (3, H, W)

            return tensor

        except Exception as e:
            if self.verbose:
                print(f"Error processing image {file_path.name}: {e}")
            return None

    def _process_video(self, file_path: Path) -> Optional[torch.Tensor]:
        """Process video file.

        Args:
            file_path: Path to video file

        Returns:
            Preprocessed video frames tensor of shape (N, 3, H, W), or None if failed
            N = number of frames extracted (may be < target_frames if some frames lack faces)
        """
        try:
            # Extract frames with face detection
            face_crops = self.video_processor.process_video(
                file_path,
                target_size=self.input_size,
            )  # Shape: (N, H, W, 3)

            if face_crops.shape[0] == 0:
                if self.verbose:
                    print(f"Warning: No faces detected in video: {file_path.name}")
                return None

            # Preprocess each frame
            frame_tensors = []
            for frame in face_crops:
                tensor = self.preprocessor(frame)  # Shape: (3, H, W)
                frame_tensors.append(tensor)

            # Stack frames
            video_tensor = torch.stack(frame_tensors, dim=0)  # Shape: (N, 3, H, W)

            return video_tensor

        except Exception as e:
            if self.verbose:
                print(f"Error processing video {file_path.name}: {e}")
            return None

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """Get dataset item.

        Args:
            idx: Index

        Returns:
            Dictionary containing:
                - "frames": Tensor of shape (N, 3, H, W)
                    - For images: N=1
                    - For videos: N=num_frames
                - "filename": File name (str)
                - "is_video": Boolean flag
                - "label": Ground truth label (only in training mode)
                    - 0 = Real
                    - 1 = Fake

        Note:
            If processing fails and fallback_strategy is "skip", frames will be
            a zero tensor of shape (1, 3, H, W).
        """
        file_path = self.file_list[idx]
        filename = file_path.name

        # Determine file type
        is_video = self._is_video(file_path)

        # Process file
        if is_video:
            frames = self._process_video(file_path)
        else:
            frames = self._process_image(file_path)

        # Handle processing failure
        if frames is None:
            # Create zero tensor as placeholder
            frames = torch.zeros(1, 3, *self.input_size)
            if self.verbose:
                print(f"Warning: Using zero tensor for {filename}")

        # For images, ensure shape is (1, 3, H, W) for consistency
        if not is_video and frames.dim() == 3:
            frames = frames.unsqueeze(0)  # (3, H, W) → (1, 3, H, W)

        # Build sample dictionary
        sample = {
            "frames": frames,
            "filename": filename,
            "is_video": is_video,
        }

        # Add label for training mode
        if self.mode == "train":
            label = self.labels_dict.get(filename, -1)
            if label == -1:
                raise ValueError(
                    f"Label not found for {filename} in labels_dict. "
                    f"All training files must have labels."
                )
            sample["label"] = label

        return sample

    def get_file_list(self) -> List[str]:
        """Get list of all filenames in dataset.

        Returns:
            List of filenames (sorted)
        """
        return [f.name for f in self.file_list]

    def get_statistics(self) -> Dict[str, int]:
        """Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_files": len(self.file_list),
            "num_images": sum(1 for f in self.file_list if self._is_image(f)),
            "num_videos": sum(1 for f in self.file_list if self._is_video(f)),
        }

        if self.mode == "train" and self.labels_dict:
            num_real = sum(1 for label in self.labels_dict.values() if label == 0)
            num_fake = sum(1 for label in self.labels_dict.values() if label == 1)
            stats["num_real"] = num_real
            stats["num_fake"] = num_fake
            stats["class_balance"] = num_fake / (num_real + num_fake) if (num_real + num_fake) > 0 else 0.0

        return stats


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Custom collate function for DataLoader.

    Handles variable-length video frames by batching them separately.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dictionary:
            - "frames": List of tensors (not stacked due to variable length)
            - "filename": List of filenames
            - "is_video": List of boolean flags
            - "label": Tensor of labels (only in training mode)
            - "num_frames": List of frame counts

    Example:
        >>> dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        >>> batch = next(iter(dataloader))
        >>> # batch["frames"] is a list of tensors, not a single tensor
        >>> # batch["frames"][0].shape might be (1, 3, 224, 224) for image
        >>> # batch["frames"][1].shape might be (16, 3, 224, 224) for video
    """
    frames_list = [sample["frames"] for sample in batch]
    filenames = [sample["filename"] for sample in batch]
    is_video_list = [sample["is_video"] for sample in batch]
    num_frames = [sample["frames"].shape[0] for sample in batch]

    batched = {
        "frames": frames_list,  # List of tensors (variable length)
        "filename": filenames,
        "is_video": is_video_list,
        "num_frames": num_frames,
    }

    # Add labels if present (training mode)
    if "label" in batch[0]:
        labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
        batched["label"] = labels

    return batched


def create_inference_dataset(
    data_dir: Union[str, Path] = "./data",
    face_detector_name: str = "mtcnn",
    video_target_frames: int = 16,
    input_size: Tuple[int, int] = (224, 224),
    device: str = "cuda",
    verbose: bool = True,
) -> DeepfakeDataset:
    """Factory function for creating inference dataset.

    Args:
        data_dir: Path to test data directory
        face_detector_name: Face detector backend
        video_target_frames: Number of frames per video
        input_size: Input image size
        device: Device for processing
        verbose: Print information

    Returns:
        DeepfakeDataset configured for inference

    Example:
        >>> dataset = create_inference_dataset("./data")
        >>> dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    """
    return DeepfakeDataset(
        data_dir=data_dir,
        mode="inference",
        labels_dict=None,
        face_detector_name=face_detector_name,
        face_confidence_threshold=0.9,
        face_fallback_strategy="center_crop",
        video_target_frames=video_target_frames,
        video_sampling_method="uniform",
        input_size=input_size,
        augmentation_config=None,
        device=device,
        verbose=verbose,
    )


def create_training_dataset(
    data_dir: Union[str, Path],
    labels_dict: Dict[str, int],
    face_detector_name: str = "mtcnn",
    video_target_frames: int = 16,
    input_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> DeepfakeDataset:
    """Factory function for creating training dataset.

    Args:
        data_dir: Path to training data directory
        labels_dict: Dictionary mapping filename → label (0=Real, 1=Fake)
        face_detector_name: Face detector backend
        video_target_frames: Number of frames per video
        input_size: Input image size
        augmentation_config: Augmentation configuration
        device: Device for processing
        verbose: Print information

    Returns:
        DeepfakeDataset configured for training

    Example:
        >>> labels = {"fake_001.jpg": 1, "real_002.mp4": 0, ...}
        >>> dataset = create_training_dataset("./data/train", labels)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    """
    return DeepfakeDataset(
        data_dir=data_dir,
        mode="train",
        labels_dict=labels_dict,
        face_detector_name=face_detector_name,
        face_confidence_threshold=0.9,
        face_fallback_strategy="center_crop",
        video_target_frames=video_target_frames,
        video_sampling_method="uniform",
        input_size=input_size,
        augmentation_config=augmentation_config,
        device=device,
        verbose=verbose,
    )
