"""Training script for deepfake detection model.

This script orchestrates the complete training pipeline:
- Configuration loading
- Dataset preparation (train/val split)
- Model initialization
- Training with early stopping
- Checkpoint management

Usage:
    python scripts/train.py --config configs/training_config.yaml --experiment my_experiment
    python scripts/train.py --config configs/training_config.yaml --resume checkpoints/best.pth

Example:
    # Train from scratch
    python scripts/train.py \\
        --config configs/training_config.yaml \\
        --model-config configs/model_config.yaml \\
        --train-data data/train \\
        --train-labels data/train_labels.csv \\
        --val-data data/val \\
        --val-labels data/val_labels.csv \\
        --experiment hybrid_model \\
        --device cuda

    # Resume training
    python scripts/train.py \\
        --config configs/training_config.yaml \\
        --resume checkpoints/hybrid_model_best.pth \\
        --device cuda
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from data import DeepfakeDataset, collate_fn, create_training_dataset
from models import create_model_from_config
from training import Trainer
from utils.config import load_config, load_yaml
from utils.logger import setup_logger, log_system_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train deepfake detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config YAML file",
    )

    # Data
    parser.add_argument(
        "--train-data",
        type=str,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--train-labels",
        type=str,
        help="Path to training labels CSV file (columns: filename, label)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        help="Path to validation data directory (optional, can use --val-split instead)",
    )
    parser.add_argument(
        "--val-labels",
        type=str,
        help="Path to validation labels CSV file",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2). Used if --val-data not provided.",
    )

    # Training
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name for logging and checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (cuda or cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Directories
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (may reduce performance)",
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (single epoch, small dataset)",
    )

    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
        deterministic: Enable deterministic mode for CUDA
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_labels_from_csv(csv_path: str) -> dict:
    """Load labels from CSV file.

    Expected CSV format:
        filename,label
        image1.jpg,0
        video1.mp4,1
        ...

    Args:
        csv_path: Path to CSV file

    Returns:
        Dictionary mapping filename → label (0=Real, 1=Fake)
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate columns
    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"CSV must have 'filename' and 'label' columns. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Validate labels
    unique_labels = df["label"].unique()
    if not all(label in [0, 1] for label in unique_labels):
        raise ValueError(
            f"Labels must be 0 (Real) or 1 (Fake). Found: {unique_labels}"
        )

    # Create dictionary
    labels_dict = dict(zip(df["filename"], df["label"]))

    print(f"Loaded {len(labels_dict)} labels from {csv_path}")
    print(f"  Real (0): {sum(1 for v in labels_dict.values() if v == 0)}")
    print(f"  Fake (1): {sum(1 for v in labels_dict.values() if v == 1)}")

    return labels_dict


def create_data_loaders(args, config):
    """Create train and validation data loaders.

    Args:
        args: Command line arguments
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get data config
    training_config = config.get("training", {})
    batch_size = args.batch_size or training_config.get("batch_size", 32)
    num_workers = training_config.get("num_workers", 8)
    pin_memory = training_config.get("pin_memory", True)

    # Get augmentation config
    augmentation_config = config.get("augmentation", {})

    # Load labels
    if args.train_labels is None:
        raise ValueError("--train-labels is required for training")

    train_labels = load_labels_from_csv(args.train_labels)

    # Check if separate validation set is provided
    if args.val_data and args.val_labels:
        # Use separate validation set
        print("\nUsing separate validation set")

        val_labels = load_labels_from_csv(args.val_labels)

        # Create train dataset
        train_dataset = create_training_dataset(
            data_dir=args.train_data,
            labels_dict=train_labels,
            augmentation_config=augmentation_config,
            verbose=True,
        )

        # Create val dataset (no augmentation)
        val_dataset = create_training_dataset(
            data_dir=args.val_data,
            labels_dict=val_labels,
            augmentation_config=None,  # No augmentation for validation
            verbose=True,
        )

    else:
        # Split training data into train/val
        print(f"\nSplitting training data with ratio {args.val_split}")

        # Create full dataset
        full_dataset = create_training_dataset(
            data_dir=args.train_data,
            labels_dict=train_labels,
            augmentation_config=augmentation_config,
            verbose=True,
        )

        # Split dataset
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        print(f"  Train size: {train_size}")
        print(f"  Val size: {val_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed, args.deterministic)

    # Setup logger
    experiment_name = args.experiment or f"experiment_{args.seed}"
    log_file = Path(args.log_dir) / f"{experiment_name}_train.log"
    logger = setup_logger("training", log_file=log_file)

    logger.info("=" * 80)
    logger.info("Deepfake Detection Model Training")
    logger.info("=" * 80)

    # Log system info
    log_system_info(logger)

    # Log arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Load configuration
    logger.info("\nLoading configuration...")
    config = load_yaml(args.config)
    model_config = load_yaml(args.model_config)

    # Override config with command line args
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    # Debug mode
    if args.debug:
        logger.warning("DEBUG MODE: Training for 1 epoch only")
        config["training"]["epochs"] = 1
        config["early_stopping"]["enabled"] = False

    # Create data loaders
    logger.info("\nPreparing datasets...")
    train_loader, val_loader = create_data_loaders(args, config)

    # Create model
    logger.info("\nInitializing model...")
    model = create_model_from_config(model_config)

    # Log model info
    from models import count_parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")

    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=experiment_name,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resuming from epoch {start_epoch}")

    # Train
    logger.info("\nStarting training...")
    logger.info("=" * 80)

    try:
        history = trainer.train(epochs=config["training"]["epochs"])

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Best validation Macro F1: {trainer.best_metric:.4f}")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        logger.info("Checkpoint saved. You can resume training with --resume")
        return 1

    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
