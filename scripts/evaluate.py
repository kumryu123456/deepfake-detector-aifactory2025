"""Evaluation script for deepfake detection model.

This script evaluates a trained model on a dataset and reports metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pth --data data/test --labels data/test_labels.csv

Example:
    # Evaluate on validation set
    python scripts/evaluate.py \\
        --checkpoint checkpoints/hybrid_model_best.pth \\
        --data data/val \\
        --labels data/val_labels.csv \\
        --device cuda

    # Cross-dataset evaluation
    python scripts/evaluate.py \\
        --checkpoint checkpoints/hybrid_model_best.pth \\
        --data data/celebdf \\
        --labels data/celebdf_labels.csv \\
        --batch-size 64 \\
        --device cuda
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import create_training_dataset, collate_fn
from models import create_model_from_config
from training.metrics import MetricsCalculator
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to evaluation data directory",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels CSV file (columns: filename, label)",
    )

    # Inference
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use mixed precision (FP16) for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (optional)",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="Path to save predictions CSV (optional)",
    )

    return parser.parse_args()


def load_labels_from_csv(csv_path: str) -> dict:
    """Load labels from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Dictionary mapping filename → label
    """
    import pandas as pd

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"CSV must have 'filename' and 'label' columns. "
            f"Found: {df.columns.tolist()}"
        )

    labels_dict = dict(zip(df["filename"], df["label"]))
    return labels_dict


def evaluate(model, data_loader, device, use_fp16=False):
    """Evaluate model on dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device for inference
        use_fp16: Use mixed precision

    Returns:
        Tuple of (predictions, labels, probabilities, filenames)
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []
    all_filenames = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")

        for batch in pbar:
            frames_list = batch["frames"]
            labels = batch.get("label")
            filenames = batch["filename"]

            if labels is None:
                raise ValueError("Labels are required for evaluation")

            labels = labels.to(device)

            batch_predictions = []
            batch_probs = []

            for frames, label in zip(frames_list, labels):
                frames = frames.to(device)

                # Forward pass
                if use_fp16:
                    with torch.cuda.amp.autocast():
                        logits = model(frames)
                        logits_mean = logits.mean(dim=0, keepdim=True)
                else:
                    logits = model(frames)
                    logits_mean = logits.mean(dim=0, keepdim=True)

                # Get predictions
                probs = torch.softmax(logits_mean, dim=1)
                pred = torch.argmax(logits_mean, dim=1)

                batch_predictions.append(pred.item())
                batch_probs.append(probs.cpu().numpy())

            all_predictions.extend(batch_predictions)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(batch_probs)
            all_filenames.extend(filenames)

    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)

    return all_predictions, all_labels, all_probs, all_filenames


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup logger
    logger = setup_logger("evaluation", file_output=False)

    logger.info("=" * 80)
    logger.info("Deepfake Detection Model Evaluation")
    logger.info("=" * 80)

    # Load checkpoint
    logger.info(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # Extract config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        model_config = config.get("model", {})
    else:
        # Try to load model config from default location
        logger.warning("No config found in checkpoint, using default model config")
        from utils.config import load_yaml
        model_config = load_yaml("configs/model_config.yaml")

    # Create model
    logger.info("Creating model...")
    model = create_model_from_config({"model": model_config})
    model = model.to(args.device)

    # Load weights
    logger.info("Loading model weights...")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Log checkpoint info
    epoch = checkpoint.get("epoch", "unknown")
    best_metric = checkpoint.get("best_metric", "unknown")
    logger.info(f"Checkpoint info:")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Best metric: {best_metric}")

    # Load labels
    logger.info(f"\nLoading labels from: {args.labels}")
    labels_dict = load_labels_from_csv(args.labels)

    # Create dataset
    logger.info(f"Loading data from: {args.data}")
    dataset = create_training_dataset(
        data_dir=args.data,
        labels_dict=labels_dict,
        augmentation_config=None,  # No augmentation for evaluation
        verbose=True,
    )

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info(f"\nEvaluation settings:")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Mixed precision: {args.use_fp16}")
    logger.info(f"  Total samples: {len(dataset)}")

    # Evaluate
    logger.info("\nRunning evaluation...")
    predictions, labels, probs, filenames = evaluate(
        model=model,
        data_loader=data_loader,
        device=args.device,
        use_fp16=args.use_fp16,
    )

    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics_calculator = MetricsCalculator()

    metrics = metrics_calculator.compute_all_metrics(
        y_true=labels,
        y_pred=predictions,
        y_probs=probs,
    )

    # Print results
    print("\n")
    metrics_calculator.print_metrics_report(metrics)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_json = {
                key: float(value) if isinstance(value, (np.floating, np.integer)) else value
                for key, value in metrics.items()
                if key != "confusion_matrix"  # Skip confusion matrix for now
            }
            json.dump(metrics_json, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    # Save predictions if requested
    if args.save_predictions:
        import pandas as pd

        pred_df = pd.DataFrame({
            "filename": filenames,
            "true_label": labels,
            "predicted_label": predictions,
            "prob_real": probs[:, 0],
            "prob_fake": probs[:, 1],
        })

        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)

        logger.info(f"Predictions saved to: {pred_path}")

    # Check performance threshold
    macro_f1 = metrics["macro_f1"]
    logger.info(f"\n{'=' * 80}")
    if macro_f1 >= 0.80:
        logger.info(f"✅ SUCCESS: Macro F1 ({macro_f1:.4f}) meets target threshold (≥0.80)")
    else:
        logger.warning(f"⚠️  WARNING: Macro F1 ({macro_f1:.4f}) below target threshold (≥0.80)")

    # Check class balance
    is_balanced = metrics_calculator.is_balanced_performance(metrics, threshold=0.1)
    if is_balanced:
        logger.info("✅ Performance is balanced across classes")
    else:
        weak_class = metrics_calculator.get_weak_class(metrics)
        logger.warning(f"⚠️  Performance imbalance detected. Weak class: {weak_class}")

    logger.info(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    exit(main())
