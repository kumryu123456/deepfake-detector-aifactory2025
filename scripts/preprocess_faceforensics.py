#!/usr/bin/env python3
"""
FaceForensics++ Data Preprocessing Script

Processes FaceForensics++ dataset:
- Extracts faces from videos (Real and Fake)
- Applies face detection (MTCNN/RetinaFace/MediaPipe)
- Saves as images for training
- Splits into train/val sets

Usage:
    python scripts/preprocess_faceforensics.py \
        --input output_folder \
        --output data/faceforensics/processed \
        --detector mtcnn \
        --max-frames 10 \
        --val-split 0.2
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import logging
import random
import traceback

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import cv2
import numpy as np
from tqdm import tqdm
from data.face_detector import FaceDetector
from data.video_processor import VideoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess FaceForensics++ dataset")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory (output_folder from download script)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "retinaface", "mediapipe"],
        help="Face detector to use"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum frames to extract per video"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (0.2 = 20%)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum videos to process (for testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for face detection (cuda or cpu)"
    )

    return parser.parse_args()


def find_videos(data_dir: Path, label: str) -> List[Tuple[Path, str]]:
    """Find all video files with their labels.

    Args:
        data_dir: Root data directory
        label: "real" or "fake"

    Returns:
        List of (video_path, label) tuples
    """
    allowed_labels = {"real", "fake"}
    if label not in allowed_labels:
        raise ValueError(
            f"Invalid label '{label}'. Expected one of: {', '.join(sorted(allowed_labels))}."
        )

    videos = []

    if label == "real":
        # Original sequences
        for seq_type in ["youtube", "actors"]:
            seq_dir = data_dir / "original_sequences" / seq_type / "raw" / "videos"
            if seq_dir.exists():
                for video_file in seq_dir.glob("*.mp4"):
                    videos.append((video_file, "real"))

    elif label == "fake":
        # Manipulated sequences
        manip_dir = data_dir / "manipulated_sequences"
        if manip_dir.exists():
            for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
                method_dir = manip_dir / method / "raw" / "videos"
                if method_dir.exists():
                    for video_file in method_dir.glob("*.mp4"):
                        videos.append((video_file, "fake"))

    return videos


def process_video(
    video_path: Path,
    label: str,
    output_dir: Path,
    face_detector: FaceDetector,
    video_processor: VideoProcessor,
    max_frames: int = 10
) -> int:
    """Process a single video and extract face images.

    Args:
        video_path: Path to video file
        label: "real" or "fake"
        output_dir: Output directory for images
        face_detector: Face detector instance
        video_processor: Video processor instance
        max_frames: Maximum frames to extract

    Returns:
        Number of faces extracted
    """
    try:
        # Extract frames uniformly
        frames = video_processor.extract_frames(
            str(video_path),
            max_frames=max_frames
        )

        if len(frames) == 0:
            return 0

        # Process each frame
        faces_extracted = 0
        video_name = video_path.stem

        for i, frame in enumerate(frames):
            try:
                # Detect and crop face
                face = face_detector.detect_and_crop(
                    frame,
                    target_size=(224, 224)
                )
            except Exception:
                # Face detection failed for this frame, skip
                continue

            # Save face image
            output_path = output_dir / f"{video_name}_frame_{i:03d}.jpg"
            success = cv2.imwrite(
                str(output_path),
                cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            )
            if not success:
                logging.warning(
                    "Failed to write frame %s for video '%s' to %s",
                    i,
                    video_path.name,
                    output_path
                )
                continue
            faces_extracted += 1

        return faces_extracted

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        logging.error(
            "Error processing %s:\n%s",
            video_path.name,
            traceback.format_exc()
        )
        return -1


def process_split(
    split_title: str,
    tqdm_desc: str,
    videos: List[Tuple[Path, str]],
    output_dir: Path,
    face_detector: FaceDetector,
    video_processor: VideoProcessor,
    max_frames: int
) -> int:
    """Process a list of videos belonging to the same dataset split."""
    print("\n" + "=" * 80)
    print(split_title)
    print("=" * 80)

    total_faces = 0
    for video_path, label in tqdm(videos, desc=tqdm_desc):
        faces = process_video(
            video_path,
            label,
            output_dir,
            face_detector,
            video_processor,
            max_frames,
        )
        total_faces += max(faces, 0)
    return total_faces


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 80)
    print("FACEFORENSICS++ DATA PREPROCESSING")
    print("=" * 80)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Face detector: {args.detector}")
    print(f"Max frames per video: {args.max_frames}")
    print(f"Validation split: {args.val_split}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Create output directories
    train_real_dir = output_dir / "train" / "real"
    train_fake_dir = output_dir / "train" / "fake"
    val_real_dir = output_dir / "val" / "real"
    val_fake_dir = output_dir / "val" / "fake"

    for dir_path in [train_real_dir, train_fake_dir, val_real_dir, val_fake_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize processors
    print("\nInitializing face detector and video processor...")
    face_detector = FaceDetector(
        detector_name=args.detector,
        device=args.device,
        confidence_threshold=0.9,
        margin_ratio=0.3,
        fallback_strategy="center_crop"
    )
    video_processor = VideoProcessor()

    # Find all videos
    print("\nScanning for videos...")
    real_videos = find_videos(input_dir, "real")
    fake_videos = find_videos(input_dir, "fake")

    print(f"\nFound videos:")
    print(f"  Real: {len(real_videos)}")
    print(f"  Fake: {len(fake_videos)}")

    # Limit for testing
    if args.max_videos:
        # Favor real videos when args.max_videos is odd to reduce class imbalance and keep evaluation/training closer to real distribution.
        half = args.max_videos // 2
        remainder = args.max_videos % 2
        real_limit = min(len(real_videos), half + remainder)
        fake_limit = min(len(fake_videos), half)
        real_videos = real_videos[:real_limit]
        fake_videos = fake_videos[:fake_limit]
        print(f"\nLimited to {args.max_videos} videos for testing")

    # Shuffle and split
    random.seed(42)
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    real_split = int(len(real_videos) * (1 - args.val_split))
    fake_split = int(len(fake_videos) * (1 - args.val_split))

    real_train = real_videos[:real_split]
    real_val = real_videos[real_split:]
    fake_train = fake_videos[:fake_split]
    fake_val = fake_videos[fake_split:]

    print(f"\nTrain/Val split:")
    print(f"  Train Real: {len(real_train)}")
    print(f"  Train Fake: {len(fake_train)}")
    print(f"  Val Real: {len(real_val)}")
    print(f"  Val Fake: {len(fake_val)}")

    # Process videos
    stats = {
        "train_real": 0,
        "train_fake": 0,
        "val_real": 0,
        "val_fake": 0
    }

    split_configs = [
        ("PROCESSING TRAIN REAL", "Train Real", real_train, train_real_dir, "train_real"),
        ("PROCESSING TRAIN FAKE", "Train Fake", fake_train, train_fake_dir, "train_fake"),
        ("PROCESSING VAL REAL", "Val Real", real_val, val_real_dir, "val_real"),
        ("PROCESSING VAL FAKE", "Val Fake", fake_val, val_fake_dir, "val_fake"),
    ]

    for title, desc, videos, out_dir, stats_key in split_configs:
        stats[stats_key] += process_split(
            split_title=title,
            tqdm_desc=desc,
            videos=videos,
            output_dir=out_dir,
            face_detector=face_detector,
            video_processor=video_processor,
            max_frames=args.max_frames,
        )

    # Print final statistics
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nExtracted faces:")
    print(f"  Train Real: {stats['train_real']} images")
    print(f"  Train Fake: {stats['train_fake']} images")
    print(f"  Val Real: {stats['val_real']} images")
    print(f"  Val Fake: {stats['val_fake']} images")
    print(f"\nTotal train: {stats['train_real'] + stats['train_fake']}")
    print(f"Total val: {stats['val_real'] + stats['val_fake']}")
    print(f"Total: {sum(stats.values())}")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
