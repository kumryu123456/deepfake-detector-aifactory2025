"""Configuration management for deepfake detection project.

This module provides utilities for loading and managing YAML configuration files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "dual_branch_hybrid"
    num_classes: int = 2
    spatial_branch: Dict[str, Any] = field(default_factory=dict)
    frequency_branch: Dict[str, Any] = field(default_factory=dict)
    fusion: Dict[str, Any] = field(default_factory=dict)
    classifier: Dict[str, Any] = field(default_factory=dict)
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    optimizer: Dict[str, Any] = field(default_factory=dict)
    scheduler: Dict[str, Any] = field(default_factory=dict)
    warmup: Dict[str, Any] = field(default_factory=dict)
    loss: Dict[str, Any] = field(default_factory=dict)
    early_stopping: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data configuration."""
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    sampling: str = "balanced"
    augmentation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    """Inference configuration."""
    device: str = "cuda"
    use_fp16: bool = True
    batch_size: int = 64
    video_batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    video: Dict[str, Any] = field(default_factory=dict)
    face_detection: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    deterministic: bool = True


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    model: Optional[ModelConfig] = None
    training: Optional[TrainingConfig] = None
    data: Optional[DataConfig] = None
    inference: Optional[InferenceConfig] = None
    validation: Optional[Dict[str, Any]] = None
    checkpoint: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    seed: int = 42


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary containing YAML content

    Raises:
        FileNotFoundError: If file does not exist
        yaml.YAMLError: If YAML is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f)
            return config_dict if config_dict is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")


def load_config(config_path: Union[str, Path], config_type: str = "all") -> Union[Config, ModelConfig, TrainingConfig, DataConfig, InferenceConfig]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file or directory containing configs
        config_type: Type of config to load ("all", "model", "training", "data", "inference")

    Returns:
        Configuration object of requested type

    Raises:
        ValueError: If config_type is invalid
        FileNotFoundError: If config file not found
    """
    config_path = Path(config_path)

    # If directory provided, load specific config from it
    if config_path.is_dir():
        config_files = {
            "model": config_path / "model_config.yaml",
            "training": config_path / "training_config.yaml",
            "inference": config_path / "inference_config.yaml",
        }

        if config_type == "all":
            # Load all configs
            model_dict = load_yaml(config_files["model"]) if config_files["model"].exists() else {}
            training_dict = load_yaml(config_files["training"]) if config_files["training"].exists() else {}
            inference_dict = load_yaml(config_files["inference"]) if config_files["inference"].exists() else {}

            return Config(
                model=ModelConfig(**model_dict.get("model", {})) if model_dict else None,
                training=TrainingConfig(**training_dict.get("training", {})) if training_dict else None,
                data=DataConfig(**training_dict.get("data", {})) if training_dict else None,
                inference=InferenceConfig(**inference_dict.get("inference", {})) if inference_dict else None,
                validation=training_dict.get("validation"),
                checkpoint=training_dict.get("checkpoint"),
                logging=training_dict.get("logging"),
                seed=training_dict.get("seed", 42),
            )
        elif config_type in config_files:
            config_dict = load_yaml(config_files[config_type])

            if config_type == "model":
                return ModelConfig(**config_dict.get("model", {}))
            elif config_type == "training":
                return Config(
                    training=TrainingConfig(**config_dict.get("training", {})),
                    data=DataConfig(**config_dict.get("data", {})),
                    validation=config_dict.get("validation"),
                    checkpoint=config_dict.get("checkpoint"),
                    logging=config_dict.get("logging"),
                    seed=config_dict.get("seed", 42),
                )
            elif config_type == "inference":
                return InferenceConfig(**config_dict.get("inference", {}))
        else:
            raise ValueError(f"Invalid config_type: {config_type}. Must be one of: all, model, training, data, inference")

    # If file provided, load it directly
    else:
        config_dict = load_yaml(config_path)

        # Determine config type from content
        if "model" in config_dict and config_type in ["all", "model"]:
            return ModelConfig(**config_dict["model"])
        elif "training" in config_dict and config_type in ["all", "training"]:
            return Config(
                training=TrainingConfig(**config_dict.get("training", {})),
                data=DataConfig(**config_dict.get("data", {})),
                validation=config_dict.get("validation"),
                checkpoint=config_dict.get("checkpoint"),
                logging=config_dict.get("logging"),
                seed=config_dict.get("seed", 42),
            )
        elif "inference" in config_dict and config_type in ["all", "inference"]:
            return InferenceConfig(**config_dict["inference"])
        else:
            # Return raw dict if structure doesn't match
            return config_dict


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: Variable number of config dictionaries

    Returns:
        Merged configuration dictionary
    """
    merged = {}

    for config in configs:
        if config is not None:
            merged = _deep_merge(merged, config)

    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Union[Config, Dict[str, Any]], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object or dictionary
        save_path: Path to save config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict if needed
    if hasattr(config, '__dataclass_fields__'):
        config_dict = _dataclass_to_dict(config)
    else:
        config_dict = config

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass to dictionary recursively.

    Args:
        obj: Dataclass object or other type

    Returns:
        Dictionary representation or original object
    """
    if hasattr(obj, '__dataclass_fields__'):
        return {
            key: _dataclass_to_dict(value)
            for key, value in obj.__dict__.items()
            if value is not None
        }
    elif isinstance(obj, dict):
        return {key: _dataclass_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    else:
        return obj
