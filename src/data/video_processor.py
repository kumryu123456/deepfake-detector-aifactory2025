"""Video frame extraction and processing module.

This module provides utilities for extracting frames from videos
and processing them for deepfake detection.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from .face_detector import FaceDetector


class VideoProcessor:
    """Video frame extraction and processing interface.

    Extracts frames from videos using various sampling strategies
    and optionally performs face detection on each frame.
    """

    def __init__(
        self,
        target_frames: int = 16,
        sampling_method: str = "uniform",
        face_detector: Optional[FaceDetector] = None,
        show_progress: bool = False,
    ):
        """Initialize video processor.

        Args:
            target_frames: Number of frames to extract per video
            sampling_method: Sampling strategy ("uniform", "random", "adaptive")
            face_detector: Face detector instance for per-frame processing
            show_progress: Show progress bar for frame extraction
        """
        self.target_frames = target_frames
        self.sampling_method = sampling_method
        self.face_detector = face_detector
        self.show_progress = show_progress

    def extract_frames(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Extract frames from video file.

        Args:
            video_path: Path to video file (.mp4)
            max_frames: Maximum frames to extract (overrides target_frames)

        Returns:
            List of RGB frames as numpy arrays (H, W, 3)
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Determine target frame count
            num_frames = max_frames if max_frames is not None else self.target_frames
            num_frames = min(num_frames, total_frames)

            # Get frame indices based on sampling method
            if self.sampling_method == "uniform":
                frame_indices = self._uniform_sampling(total_frames, num_frames)
            elif self.sampling_method == "random":
                frame_indices = self._random_sampling(total_frames, num_frames)
            elif self.sampling_method == "adaptive":
                frame_indices = self._adaptive_sampling(total_frames, num_frames, fps)
            else:
                # Default to uniform
                frame_indices = self._uniform_sampling(total_frames, num_frames)

            # Extract frames
            frames = []
            frame_iter = tqdm(frame_indices, desc="Extracting frames") if self.show_progress else frame_indices

            for frame_idx in frame_iter:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                # Read frame
                ret, frame = cap.read()

                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            return frames

        finally:
            cap.release()

    def _uniform_sampling(self, total_frames: int, target_frames: int) -> List[int]:
        """Uniform sampling: evenly spaced frames across video.

        Args:
            total_frames: Total number of frames in video
            target_frames: Number of frames to sample

        Returns:
            List of frame indices
        """
        if total_frames <= target_frames:
            return list(range(total_frames))

        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        return indices.tolist()

    def _random_sampling(self, total_frames: int, target_frames: int) -> List[int]:
        """Random sampling: randomly selected frames.

        Args:
            total_frames: Total number of frames in video
            target_frames: Number of frames to sample

        Returns:
            List of frame indices (sorted)
        """
        if total_frames <= target_frames:
            return list(range(total_frames))

        indices = np.random.choice(total_frames, target_frames, replace=False)
        return sorted(indices.tolist())

    def _adaptive_sampling(
        self,
        total_frames: int,
        target_frames: int,
        fps: float,
    ) -> List[int]:
        """Adaptive sampling: more frames from early part of video.

        For 5-second videos, early frames often contain more information.

        Args:
            total_frames: Total number of frames in video
            target_frames: Number of frames to sample
            fps: Video frame rate

        Returns:
            List of frame indices
        """
        if total_frames <= target_frames:
            return list(range(total_frames))

        # Sample more from first half
        first_half_count = int(target_frames * 0.6)
        second_half_count = target_frames - first_half_count

        mid_point = total_frames // 2

        # Sample from first half
        first_half_indices = np.linspace(0, mid_point - 1, first_half_count, dtype=int)

        # Sample from second half
        second_half_indices = np.linspace(mid_point, total_frames - 1, second_half_count, dtype=int)

        # Combine and sort
        indices = np.concatenate([first_half_indices, second_half_indices])
        return sorted(indices.tolist())

    def process_video(
        self,
        video_path: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Extract frames and detect faces.

        Args:
            video_path: Path to video file
            target_size: Size for cropped faces

        Returns:
            Array of face crops, shape (num_frames, H, W, 3)
            num_frames may be < target_frames if some frames lack faces

        Raises:
            ValueError: If no face detector is configured
        """
        if self.face_detector is None:
            raise ValueError("Face detector must be provided for process_video()")

        # Extract frames
        frames = self.extract_frames(video_path)

        # Detect faces in each frame
        face_crops = []

        for frame in frames:
            try:
                face = self.face_detector.detect_and_crop(frame, target_size)

                if face is not None:
                    face_crops.append(face)

            except Exception as e:
                # Skip frames with detection errors
                continue

        # Convert to numpy array
        if len(face_crops) > 0:
            return np.stack(face_crops, axis=0)
        else:
            # If no faces detected, return empty array
            return np.zeros((0, target_size[0], target_size[1], 3), dtype=np.uint8)

    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """Get video metadata.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information:
                - total_frames: Number of frames
                - fps: Frame rate
                - duration: Duration in seconds
                - width: Frame width
                - height: Frame height
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration = total_frames / fps if fps > 0 else 0

            return {
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
                "width": width,
                "height": height,
            }

        finally:
            cap.release()

    def extract_single_frame(
        self,
        video_path: Union[str, Path],
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Extract a single frame from video.

        Args:
            video_path: Path to video file
            frame_idx: Frame index to extract (0-based)

        Returns:
            RGB frame as numpy array (H, W, 3)
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Read frame
            ret, frame = cap.read()

            if not ret:
                raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return frame_rgb

        finally:
            cap.release()

    def batch_process_videos(
        self,
        video_paths: List[Union[str, Path]],
        target_size: Tuple[int, int] = (224, 224),
    ) -> List[np.ndarray]:
        """Process batch of videos.

        Args:
            video_paths: List of video file paths
            target_size: Size for cropped faces

        Returns:
            List of face crop arrays, one per video
        """
        results = []

        video_iter = tqdm(video_paths, desc="Processing videos") if self.show_progress else video_paths

        for video_path in video_iter:
            try:
                face_crops = self.process_video(video_path, target_size)
                results.append(face_crops)
            except Exception as e:
                # Append empty array for failed videos
                results.append(np.zeros((0, target_size[0], target_size[1], 3), dtype=np.uint8))

        return results
