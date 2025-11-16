"""Model checkpoint loading utilities for inference.

This module provides utilities for loading trained models from checkpoints
with proper configuration handling and device placement.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml


class ModelLoader:
    """Utility class for loading model checkpoints.

    Provides static methods for loading checkpoints with automatic
    configuration extraction and device placement.

    Example:
        >>> model, config = ModelLoader.load_checkpoint(
        ...     "checkpoints/best.pth",
        ...     device="cuda"
        ... )
        >>> model.eval()
    """

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        model_class: Optional[type] = None,
        strict: bool = True,
    ) -> Tuple[nn.Module, Dict]:
        """Load model from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.pth)
            device: Device to load model on ("cuda" or "cpu")
            model_class: Model class to instantiate (if None, uses config)
            strict: Whether to strictly enforce state_dict keys match

        Returns:
            Tuple of (model, config_dict)

        Raises:
            FileNotFoundError: If checkpoint file not found
            RuntimeError: If checkpoint format is invalid

        Example:
            >>> model, config = ModelLoader.load_checkpoint(
            ...     "checkpoints/hybrid_best.pth",
            ...     device="cuda"
            ... )
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract config from checkpoint
        if "config" not in checkpoint:
            raise RuntimeError(
                f"Checkpoint missing 'config' key. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        config = checkpoint["config"]

        # Create model from config
        if model_class is None:
            # Use factory function from models module
            from models import create_model_from_config

            model_config = config.get("model", {})
            model = create_model_from_config({"model": model_config})
        else:
            # Use provided model class
            model = model_class(**config.get("model", {}))

        # Load state dict
        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                f"Checkpoint missing 'model_state_dict' key. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Move to device
        model = model.to(device)

        # Set to eval mode by default
        model.eval()

        return model, config

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML is invalid

        Example:
            >>> config = ModelLoader.load_config("configs/model_config.yaml")
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config if config is not None else {}

    @staticmethod
    def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict:
        """Get information about checkpoint without loading model.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with checkpoint information:
                - epoch: Training epoch
                - best_metric: Best validation metric
                - config: Model configuration
                - timestamp: Checkpoint creation time (if available)

        Example:
            >>> info = ModelLoader.get_checkpoint_info("checkpoints/best.pth")
            >>> print(f"Epoch: {info['epoch']}, F1: {info['best_metric']:.4f}")
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        info = {
            "epoch": checkpoint.get("epoch", "unknown"),
            "best_metric": checkpoint.get("best_metric", "unknown"),
            "config": checkpoint.get("config", {}),
        }

        # Add file metadata
        import os
        stat = os.stat(checkpoint_path)
        import time
        info["timestamp"] = time.ctime(stat.st_mtime)
        info["file_size_mb"] = stat.st_size / (1024 * 1024)

        return info

    @staticmethod
    def load_model_from_config(
        model_config_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ) -> nn.Module:
        """Load model from config file, optionally loading weights from checkpoint.

        Args:
            model_config_path: Path to model config YAML file
            checkpoint_path: Optional path to checkpoint for weights
            device: Device to load model on

        Returns:
            Model instance

        Example:
            >>> # Load model architecture only
            >>> model = ModelLoader.load_model_from_config("configs/model_config.yaml")
            >>>
            >>> # Load architecture + weights
            >>> model = ModelLoader.load_model_from_config(
            ...     "configs/model_config.yaml",
            ...     checkpoint_path="checkpoints/best.pth"
            ... )
        """
        # Load config
        config = ModelLoader.load_config(model_config_path)

        # Create model
        from models import create_model_from_config
        model = create_model_from_config(config)

        # Load weights if checkpoint provided
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=device)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Try direct state dict
                model.load_state_dict(checkpoint)

        # Move to device and set eval mode
        model = model.to(device)
        model.eval()

        return model

    @staticmethod
    def verify_checkpoint_compatibility(
        checkpoint_path: Union[str, Path],
        expected_architecture: Optional[str] = None,
    ) -> bool:
        """Verify checkpoint compatibility before loading.

        Args:
            checkpoint_path: Path to checkpoint file
            expected_architecture: Expected model architecture name (optional)

        Returns:
            True if checkpoint is compatible, False otherwise

        Example:
            >>> is_compatible = ModelLoader.verify_checkpoint_compatibility(
            ...     "checkpoints/best.pth",
            ...     expected_architecture="dual_branch_hybrid"
            ... )
        """
        try:
            info = ModelLoader.get_checkpoint_info(checkpoint_path)

            # Check if config exists
            if not info["config"]:
                return False

            # Check architecture if specified
            if expected_architecture is not None:
                model_config = info["config"].get("model", {})
                architecture = model_config.get("architecture")

                if architecture != expected_architecture:
                    return False

            return True

        except Exception:
            return False


def load_checkpoint(checkpoint_path: Union[str, Path], device: str = "cuda") -> Tuple[nn.Module, Dict]:
    """Convenience function for loading checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        Tuple of (model, config)

    Example:
        >>> model, config = load_checkpoint("checkpoints/best.pth")
    """
    return ModelLoader.load_checkpoint(checkpoint_path, device)


def load_model_from_config(
    model_config_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
) -> nn.Module:
    """Convenience function for loading model from config.

    Args:
        model_config_path: Path to model config
        checkpoint_path: Optional checkpoint path
        device: Device to load on

    Returns:
        Model instance

    Example:
        >>> model = load_model_from_config("configs/model_config.yaml")
    """
    return ModelLoader.load_model_from_config(model_config_path, checkpoint_path, device)
