"""Data preprocessing and augmentation transforms.

This module provides data preprocessing pipelines for training and inference,
including data augmentation strategies based on research findings.
"""

from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image


class DataPreprocessor:
    """Data preprocessing and augmentation pipeline.

    Handles image preprocessing for both training (with augmentation) and
    inference (minimal preprocessing).
    """

    def __init__(
        self,
        mode: str = "inference",
        input_size: Tuple[int, int] = (224, 224),
        augmentation_config: Optional[Dict] = None,
    ):
        """Initialize data preprocessor.

        Args:
            mode: "train" for augmentation, "inference" for minimal preprocessing
            input_size: Target image size (height, width)
            augmentation_config: Augmentation parameters (for training mode)
        """
        self.mode = mode
        self.input_size = input_size
        self.augmentation_config = augmentation_config or {}

        # Build transform pipeline
        if mode == "train":
            self.transform = self._build_train_transforms()
        else:
            self.transform = self._build_inference_transforms()

    def _build_inference_transforms(self) -> A.Compose:
        """Build inference preprocessing pipeline (minimal transforms).

        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            A.Resize(height=self.input_size[0], width=self.input_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def _build_train_transforms(self) -> A.Compose:
        """Build training preprocessing pipeline with augmentation.

        Implements research-backed augmentation strategies for deepfake detection:
        - Geometric transformations
        - Color/appearance augmentation
        - Compression and noise (critical for generalization)

        Returns:
            Albumentations Compose object
        """
        aug_config = self.augmentation_config

        # Get augmentation parameters with defaults
        resize = aug_config.get("resize", [256, 256])
        crop_size = aug_config.get("random_crop", [224, 224])

        # Geometric augmentations
        h_flip_p = aug_config.get("horizontal_flip", 0.5)
        rotation = aug_config.get("rotation_degrees", 15)

        # Color augmentations
        color_jitter = aug_config.get("color_jitter", {})
        brightness = color_jitter.get("brightness", 0.2)
        contrast = color_jitter.get("contrast", 0.2)
        saturation = color_jitter.get("saturation", 0.2)
        hue = color_jitter.get("hue", 0.1)
        color_p = color_jitter.get("p", 0.5)

        # Noise and blur
        blur_config = aug_config.get("gaussian_blur", {})
        blur_kernel = blur_config.get("kernel_size", 5)
        blur_p = blur_config.get("p", 0.05)

        noise_config = aug_config.get("gaussian_noise", {})
        noise_sigma = noise_config.get("sigma", 0.05)
        noise_p = noise_config.get("p", 0.1)

        # Compression (important for robustness)
        compression_config = aug_config.get("jpeg_compression", {})
        quality_range = compression_config.get("quality_range", [60, 100])
        compression_p = compression_config.get("p", 0.5)

        transforms_list = [
            # Resize
            A.Resize(height=resize[0], width=resize[1]),

            # Geometric transformations
            A.HorizontalFlip(p=h_flip_p),
            A.Rotate(limit=rotation, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=rotation,
                p=0.5,
            ),

            # Random crop
            A.RandomCrop(height=crop_size[0], width=crop_size[1]),

            # Color augmentations
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=color_p,
            ),

            # Blur (one of multiple options)
            A.OneOf([
                A.GaussianBlur(blur_limit=blur_kernel, p=1.0),
                A.MedianBlur(blur_limit=blur_kernel, p=1.0),
                A.MotionBlur(blur_limit=blur_kernel, p=1.0),
            ], p=blur_p),

            # Distortions (subtle)
            A.OneOf([
                A.OpticalDistortion(p=1.0),
                A.GridDistortion(p=1.0),
                A.ElasticTransform(p=1.0),
            ], p=0.1),

            # Compression and noise (critical for generalization)
            A.ImageCompression(
                quality_lower=quality_range[0],
                quality_upper=quality_range[1],
                compression_type=A.ImageCompression.ImageCompressionType.JPEG,
                p=compression_p,
            ),
            A.GaussNoise(
                var_limit=(noise_sigma * 255) ** 2,
                p=noise_p,
            ),

            # Normalization and tensor conversion
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]

        return A.Compose(transforms_list)

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Apply transforms to image.

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            Preprocessed image tensor of shape (C, H, W)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Apply transforms
        transformed = self.transform(image=image)
        return transformed["image"]

    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess batch of images.

        Args:
            images: List of RGB images as numpy arrays

        Returns:
            Batched tensor of shape (B, C, H, W)
        """
        batch = [self(img) for img in images]
        return torch.stack(batch, dim=0)


def get_train_transforms(
    input_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None,
) -> A.Compose:
    """Get training transforms.

    Args:
        input_size: Target image size
        augmentation_config: Augmentation configuration

    Returns:
        Albumentations Compose object
    """
    preprocessor = DataPreprocessor(
        mode="train",
        input_size=input_size,
        augmentation_config=augmentation_config,
    )
    return preprocessor.transform


def get_inference_transforms(
    input_size: Tuple[int, int] = (224, 224),
) -> A.Compose:
    """Get inference transforms.

    Args:
        input_size: Target image size

    Returns:
        Albumentations Compose object
    """
    preprocessor = DataPreprocessor(
        mode="inference",
        input_size=input_size,
    )
    return preprocessor.transform


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """Denormalize image tensor for visualization.

    Args:
        tensor: Normalized image tensor of shape (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized image as numpy array (H, W, C) in range [0, 255]
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image in batch

    # Clone to avoid in-place modification
    tensor = tensor.clone().detach().cpu()

    # Denormalize
    for i, (m, s) in enumerate(zip(mean, std)):
        tensor[i] = tensor[i] * s + m

    # Clip to valid range
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy and rearrange dimensions
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    return image


def apply_center_crop(
    image: np.ndarray,
    crop_size: Tuple[int, int],
) -> np.ndarray:
    """Apply center crop to image (fallback for face detection failure).

    Args:
        image: Input image (H, W, C)
        crop_size: Crop size (height, width)

    Returns:
        Center-cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Calculate crop coordinates
    start_h = max(0, (h - crop_h) // 2)
    start_w = max(0, (w - crop_w) // 2)
    end_h = start_h + crop_h
    end_w = start_w + crop_w

    # Crop
    cropped = image[start_h:end_h, start_w:end_w]

    # Resize if needed
    if cropped.shape[:2] != crop_size:
        cropped = cv2.resize(cropped, (crop_w, crop_h))

    return cropped
