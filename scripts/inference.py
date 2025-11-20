"""Inference script for deepfake detection competition submission.

This script runs inference on test data and generates submission.csv.

Usage:
    python scripts/inference.py --checkpoint checkpoints/best.pth --data ./data --output submission.csv

Example:
    # Basic usage
    python scripts/inference.py \\
        --checkpoint checkpoints/hybrid_best.pth \\
        --data ./data \\
        --output submission.csv

    # With FP16 and custom batch size
    python scripts/inference.py \\
        --checkpoint checkpoints/hybrid_best.pth \\
        --data ./data \\
        --output submission.csv \\
        --use-fp16 \\
        --batch-size 64 \\
        --device cuda
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import torch

from inference import create_inference_engine
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on test data for competition submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="./data",
        help="Path to test data directory (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Path to save submission.csv (default: submission.csv)",
    )

    # Inference settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use mixed precision (FP16) for faster inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for image inference (default: 64)",
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        default=16,
        help="Number of frames to extract per video (default: 16)",
    )

    # Face detection
    parser.add_argument(
        "--face-detector",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "retinaface", "mediapipe"],
        help="Face detector backend (default: mtcnn)",
    )

    # Output options
    parser.add_argument(
        "--save-stats",
        type=str,
        default=None,
        help="Save inference statistics to JSON file (optional)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate submission.csv format after generation",
    )

    # Debug
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def validate_submission(output_path: Path) -> bool:
    """Validate submission.csv format.

    Args:
        output_path: Path to submission.csv

    Returns:
        True if valid, False otherwise
    """
    import pandas as pd

    try:
        df = pd.read_csv(output_path)

        # Check columns
        if list(df.columns) != ["filename", "label"]:
            print(f"❌ Invalid columns: {df.columns.tolist()}")
            print(f"   Expected: ['filename', 'label']")
            return False

        # Check for null values
        if df.isnull().any().any():
            print("❌ Found null values in submission")
            return False

        # Check label values
        unique_labels = df["label"].unique()
        if not all(label in [0, 1] for label in unique_labels):
            print(f"❌ Invalid label values: {unique_labels}")
            print(f"   Expected: only 0 and 1")
            return False

        # Check filenames have extensions
        if not all("." in filename for filename in df["filename"]):
            print("❌ Some filenames missing extensions")
            return False

        print("✅ Submission format validation passed")
        print(f"   Total predictions: {len(df)}")
        print(f"   Real (0): {sum(df['label'] == 0)}")
        print(f"   Fake (1): {sum(df['label'] == 1)}")

        return True

    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def main():
    """Main inference function."""
    args = parse_args()

    # Setup logger
    logger = setup_logger("inference", file_output=False)

    logger.info("=" * 80)
    logger.info("Deepfake Detection Inference")
    logger.info("=" * 80)

    # Log arguments
    logger.info("Configuration:")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Data directory: {args.data}")
    logger.info(f"  Output path: {args.output}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Mixed precision (FP16): {args.use_fp16}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Video frames: {args.video_frames}")
    logger.info(f"  Face detector: {args.face_detector}")

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    # Validate data directory exists
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    # Create inference engine
    logger.info("\nInitializing inference engine...")
    try:
        engine = create_inference_engine(
            checkpoint_path=checkpoint_path,
            device=args.device,
            use_fp16=args.use_fp16,
            batch_size=args.batch_size,
            video_frames=args.video_frames,
            face_detector_name=args.face_detector,
            verbose=not args.quiet,
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Get data statistics
    logger.info("\nScanning test data...")
    stats = engine.get_statistics(data_dir)
    logger.info(f"Found {stats['total_files']} files:")
    logger.info(f"  Images: {stats['num_images']}")
    logger.info(f"  Videos: {stats['num_videos']}")

    # Run inference
    logger.info("\nRunning inference...")
    start_time = time.time()

    try:
        results_df = engine.run_inference(
            data_dir=data_dir,
            output_path=args.output,
        )

        elapsed_time = time.time() - start_time

        logger.info("\n" + "=" * 80)
        logger.info("Inference completed successfully!")
        logger.info(f"  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"  Average time per file: {elapsed_time / len(results_df):.3f} seconds")
        logger.info(f"  Output saved to: {args.output}")
        logger.info("=" * 80)

        # Save statistics if requested
        if args.save_stats:
            import json
            stats_path = Path(args.save_stats)
            stats_path.parent.mkdir(parents=True, exist_ok=True)

            stats_data = {
                "total_files": len(results_df),
                "num_real": int(sum(results_df["label"] == 0)),
                "num_fake": int(sum(results_df["label"] == 1)),
                "elapsed_time": elapsed_time,
                "avg_time_per_file": elapsed_time / len(results_df),
                "device": args.device,
                "use_fp16": args.use_fp16,
                "batch_size": args.batch_size,
            }

            with open(stats_path, 'w') as f:
                json.dump(stats_data, f, indent=2)

            logger.info(f"Statistics saved to: {stats_path}")

        # Validate submission if requested
        if args.validate:
            logger.info("\nValidating submission format...")
            is_valid = validate_submission(Path(args.output))

            if not is_valid:
                logger.warning("Submission validation failed")
                return 1

        return 0

    except Exception as e:
        logger.error(f"\nInference failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
