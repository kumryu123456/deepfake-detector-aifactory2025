"""Inference engine for deepfake detection competition submission.

This module implements the complete inference pipeline for processing
test data and generating submission.csv in competition format.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from data import FaceDetector, VideoProcessor, DataPreprocessor


class InferenceEngine:
    """Inference engine for competition submission.

    Handles complete inference pipeline:
    - Load test data from ./data/
    - Process images and videos
    - Generate predictions
    - Save to submission.csv

    Example:
        >>> engine = InferenceEngine(
        ...     model=model,
        ...     device="cuda",
        ...     use_fp16=True,
        ... )
        >>> engine.run_inference(
        ...     data_dir="./data",
        ...     output_path="submission.csv"
        ... )
    """

    # Supported file extensions (case-insensitive)
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    VIDEO_EXTENSIONS = {".mp4"}

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 64,
        video_batch_size: int = 16,
        video_frames: int = 16,
        face_detector_name: str = "mtcnn",
        input_size: Tuple[int, int] = (224, 224),
        verbose: bool = True,
    ):
        """Initialize inference engine.

        Args:
            model: Trained PyTorch model
            device: Device for inference ("cuda" or "cpu")
            use_fp16: Use mixed precision (FP16) for speed
            batch_size: Batch size for image inference
            video_batch_size: Batch size for video frame processing
            video_frames: Number of frames to extract per video
            face_detector_name: Face detector backend
            input_size: Input image size (height, width)
            verbose: Print progress information
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.video_batch_size = video_batch_size
        self.video_frames = video_frames
        self.input_size = input_size
        self.verbose = verbose

        # Initialize face detector
        self.face_detector = FaceDetector(
            detector_name=face_detector_name,
            confidence_threshold=0.9,
            margin_ratio=0.3,
            device=device,
            fallback_strategy="center_crop",
        )

        # Initialize video processor
        self.video_processor = VideoProcessor(
            target_frames=video_frames,
            sampling_method="uniform",
            face_detector=self.face_detector,
            show_progress=False,
        )

        # Initialize preprocessor (inference mode, no augmentation)
        self.preprocessor = DataPreprocessor(
            mode="inference",
            input_size=input_size,
            augmentation_config=None,
        )

        if self.verbose:
            print("InferenceEngine initialized:")
            print(f"  Device: {self.device}")
            print(f"  Mixed precision (FP16): {self.use_fp16}")
            print(f"  Image batch size: {self.batch_size}")
            print(f"  Video frames: {self.video_frames}")

    def run_inference(
        self,
        data_dir: Union[str, Path] = "./data",
        output_path: Union[str, Path] = "submission.csv",
    ) -> pd.DataFrame:
        """Run complete inference pipeline.

        Args:
            data_dir: Directory containing test data (mixed images/videos)
            output_path: Path to save submission.csv

        Returns:
            DataFrame with columns [filename, label]

        Example:
            >>> results = engine.run_inference(
            ...     data_dir="./data",
            ...     output_path="submission.csv"
            ... )
        """
        data_dir = Path(data_dir)
        output_path = Path(output_path)

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Scan for files
        files = self._scan_directory(data_dir)

        if len(files) == 0:
            raise ValueError(f"No supported files found in {data_dir}")

        if self.verbose:
            print(f"\nFound {len(files)} files:")
            print(f"  Images: {sum(1 for f in files if self._is_image(f))}")
            print(f"  Videos: {sum(1 for f in files if self._is_video(f))}")

        # Process all files
        start_time = time.time()
        results = []

        for file_path in tqdm(files, desc="Processing", disable=not self.verbose):
            try:
                if self._is_image(file_path):
                    prediction = self.process_image(file_path)
                else:
                    prediction = self.process_video(file_path)

                results.append({
                    "filename": file_path.name,
                    "label": prediction,
                })

            except Exception as e:
                if self.verbose:
                    print(f"Error processing {file_path.name}: {e}")
                # Default to 0 (Real) for failed files
                results.append({
                    "filename": file_path.name,
                    "label": 0,
                })

        elapsed_time = time.time() - start_time

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        if self.verbose:
            print(f"\nInference complete:")
            print(f"  Total files: {len(results)}")
            print(f"  Real (0): {sum(1 for r in results if r['label'] == 0)}")
            print(f"  Fake (1): {sum(1 for r in results if r['label'] == 1)}")
            print(f"  Time elapsed: {elapsed_time:.2f} seconds")
            print(f"  Saved to: {output_path}")

        return df

    def process_image(self, image_path: Union[str, Path]) -> int:
        """Process single image file.

        Args:
            image_path: Path to image file

        Returns:
            Prediction label (0=Real, 1=Fake)
        """
        image_path = Path(image_path)

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return 0  # Default to Real if read fails

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect and crop face
        face = self.face_detector.detect_and_crop(image, target_size=self.input_size)

        if face is None:
            return 0  # Default to Real if no face detected

        # Preprocess
        tensor = self.preprocessor(face)  # Shape: (3, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)  # Shape: (1, 3, H, W)

        # Inference
        with torch.no_grad():
            if self.use_fp16:
                with autocast():
                    logits = self.model(tensor)  # Shape: (1, 2)
            else:
                logits = self.model(tensor)

            # Get prediction
            pred = torch.argmax(logits, dim=1).item()

        return int(pred)

    def process_video(self, video_path: Union[str, Path]) -> int:
        """Process single video file.

        Extracts frames, processes each frame, and aggregates predictions.

        Args:
            video_path: Path to video file

        Returns:
            Prediction label (0=Real, 1=Fake)
        """
        video_path = Path(video_path)

        # Extract frames with face detection
        face_crops = self.video_processor.process_video(
            video_path,
            target_size=self.input_size,
        )  # Shape: (N, H, W, 3)

        if face_crops.shape[0] == 0:
            return 0  # Default to Real if no faces detected

        # Preprocess frames
        frame_tensors = []
        for frame in face_crops:
            tensor = self.preprocessor(frame)  # Shape: (3, H, W)
            frame_tensors.append(tensor)

        # Stack frames
        frames_batch = torch.stack(frame_tensors, dim=0).to(self.device)  # Shape: (N, 3, H, W)

        # Inference on frames
        with torch.no_grad():
            if self.use_fp16:
                with autocast():
                    logits = self.model(frames_batch)  # Shape: (N, 2)
            else:
                logits = self.model(frames_batch)

            # Aggregate predictions (mean pooling of logits)
            logits_mean = logits.mean(dim=0, keepdim=True)  # Shape: (1, 2)
            pred = torch.argmax(logits_mean, dim=1).item()

        return int(pred)

    def aggregate_frame_predictions(
        self,
        frame_logits: torch.Tensor,
        method: str = "mean",
    ) -> int:
        """Aggregate frame-level predictions into video-level prediction.

        Args:
            frame_logits: Frame predictions, shape (N, 2)
            method: Aggregation method ("mean", "max", "vote")

        Returns:
            Video-level prediction (0 or 1)
        """
        if method == "mean":
            # Mean pooling of logits
            logits_mean = frame_logits.mean(dim=0)
            pred = torch.argmax(logits_mean).item()

        elif method == "max":
            # Max pooling (most confident prediction)
            probs = torch.softmax(frame_logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            most_confident_idx = torch.argmax(max_probs)
            pred = preds[most_confident_idx].item()

        elif method == "vote":
            # Majority voting
            preds = torch.argmax(frame_logits, dim=1)
            pred = torch.mode(preds).values.item()

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return int(pred)

    def _scan_directory(self, data_dir: Path) -> List[Path]:
        """Scan directory for supported files.

        Args:
            data_dir: Directory to scan

        Returns:
            List of file paths (sorted)
        """
        files = []

        for file_path in data_dir.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()

                if ext in self.IMAGE_EXTENSIONS or ext in self.VIDEO_EXTENSIONS:
                    files.append(file_path)

        # Sort for consistency
        files = sorted(files, key=lambda x: x.name)

        return files

    def _is_image(self, file_path: Path) -> bool:
        """Check if file is an image."""
        return file_path.suffix.lower() in self.IMAGE_EXTENSIONS

    def _is_video(self, file_path: Path) -> bool:
        """Check if file is a video."""
        return file_path.suffix.lower() in self.VIDEO_EXTENSIONS

    def get_statistics(self, data_dir: Union[str, Path] = "./data") -> Dict:
        """Get statistics about test data.

        Args:
            data_dir: Data directory

        Returns:
            Dictionary with statistics
        """
        data_dir = Path(data_dir)
        files = self._scan_directory(data_dir)

        stats = {
            "total_files": len(files),
            "num_images": sum(1 for f in files if self._is_image(f)),
            "num_videos": sum(1 for f in files if self._is_video(f)),
            "file_list": [f.name for f in files],
        }

        return stats


def create_inference_engine(
    checkpoint_path: Union[str, Path],
    device: str = "cuda",
    use_fp16: bool = True,
    **kwargs,
) -> InferenceEngine:
    """Factory function for creating inference engine from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for inference
        use_fp16: Use FP16 mixed precision
        **kwargs: Additional arguments for InferenceEngine

    Returns:
        InferenceEngine instance

    Example:
        >>> engine = create_inference_engine(
        ...     "checkpoints/best.pth",
        ...     device="cuda",
        ...     use_fp16=True
        ... )
        >>> engine.run_inference("./data", "submission.csv")
    """
    from .model_loader import load_checkpoint

    # Load model
    model, config = load_checkpoint(checkpoint_path, device=device)

    # Create engine
    engine = InferenceEngine(
        model=model,
        device=device,
        use_fp16=use_fp16,
        **kwargs,
    )

    return engine
