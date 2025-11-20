"""Face detection module for deepfake detection.

This module provides face detection capabilities using MTCNN or RetinaFace,
with fallback strategies for handling detection failures.
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


class FaceNotDetectedError(Exception):
    """Raised when no face is detected in an image."""
    pass


class FaceDetector:
    """Face detection and cropping interface.

    Supports multiple detection backends (MTCNN, RetinaFace) with
    fallback strategies for robustness.
    """

    def __init__(
        self,
        detector_name: str = "mtcnn",
        confidence_threshold: float = 0.9,
        margin_ratio: float = 0.3,
        device: str = "cuda",
        fallback_strategy: str = "center_crop",
    ):
        """Initialize face detector.

        Args:
            detector_name: Detector backend ("mtcnn", "retinaface", "mediapipe")
            confidence_threshold: Minimum confidence for detection
            margin_ratio: Margin ratio around detected face box
            device: Device for computation ("cuda" or "cpu")
            fallback_strategy: Strategy when no face detected
                              ("center_crop", "skip", "error")
        """
        self.detector_name = detector_name.lower()
        self.confidence_threshold = confidence_threshold
        self.margin_ratio = margin_ratio
        self.device = device
        self.fallback_strategy = fallback_strategy

        # Initialize detector
        self.detector = self._init_detector()

    def _init_detector(self):
        """Initialize the face detector backend.

        Returns:
            Detector instance

        Raises:
            ValueError: If detector name is invalid
            ImportError: If required library is not installed
        """
        if self.detector_name == "mtcnn":
            try:
                from facenet_pytorch import MTCNN

                detector = MTCNN(
                    keep_all=False,  # Only keep best face
                    device=self.device,
                    min_face_size=80,
                    thresholds=[0.6, 0.7, 0.7],  # Three-stage thresholds
                )
                return detector
            except ImportError:
                raise ImportError(
                    "facenet-pytorch is required for MTCNN. "
                    "Install with: pip install facenet-pytorch"
                )

        elif self.detector_name == "retinaface":
            try:
                from retinaface import RetinaFace

                # RetinaFace doesn't need initialization
                return RetinaFace
            except ImportError:
                raise ImportError(
                    "retinaface-pytorch is required. "
                    "Install with: pip install retinaface-pytorch"
                )

        elif self.detector_name == "mediapipe":
            try:
                import mediapipe as mp

                detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # 1 for better distant faces
                    min_detection_confidence=self.confidence_threshold,
                )
                return detector
            except ImportError:
                raise ImportError(
                    "mediapipe is required. "
                    "Install with: pip install mediapipe"
                )

        else:
            raise ValueError(
                f"Unknown detector: {self.detector_name}. "
                f"Must be one of: mtcnn, retinaface, mediapipe"
            )

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect all faces in an image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            List of detection dictionaries, each containing:
                - "box": [x1, y1, x2, y2] coordinates
                - "confidence": Detection confidence score
                - "landmarks": Optional facial landmarks
        """
        if self.detector_name == "mtcnn":
            return self._detect_mtcnn(image)
        elif self.detector_name == "retinaface":
            return self._detect_retinaface(image)
        elif self.detector_name == "mediapipe":
            return self._detect_mediapipe(image)

    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN.

        Args:
            image: RGB image

        Returns:
            List of detections
        """
        # Convert to PIL for MTCNN
        pil_image = Image.fromarray(image)

        # Detect
        boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)

        detections = []

        if boxes is not None:
            for box, prob, landmark in zip(boxes, probs, landmarks):
                if prob >= self.confidence_threshold:
                    detections.append({
                        "box": box.tolist(),  # [x1, y1, x2, y2]
                        "confidence": float(prob),
                        "landmarks": landmark.tolist() if landmark is not None else None,
                    })

        return detections

    def _detect_retinaface(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using RetinaFace.

        Args:
            image: RGB image

        Returns:
            List of detections
        """
        # RetinaFace expects BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detect
        faces = self.detector.detect_faces(bgr_image)

        detections = []

        for key, face_info in faces.items():
            confidence = face_info["score"]

            if confidence >= self.confidence_threshold:
                facial_area = face_info["facial_area"]
                landmarks = face_info.get("landmarks", {})

                detections.append({
                    "box": facial_area,  # [x1, y1, x2, y2]
                    "confidence": float(confidence),
                    "landmarks": landmarks,
                })

        return detections

    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe.

        Args:
            image: RGB image

        Returns:
            List of detections
        """
        # Process
        results = self.detector.process(image)

        detections = []

        if results.detections:
            h, w = image.shape[:2]

            for detection in results.detections:
                confidence = detection.score[0]

                if confidence >= self.confidence_threshold:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": float(confidence),
                        "landmarks": None,
                    })

        return detections

    def crop_face(
        self,
        image: np.ndarray,
        box: List[int],
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Crop and resize face region with margin.

        Args:
            image: RGB image
            box: Face bounding box [x1, y1, x2, y2]
            target_size: Output size (height, width)

        Returns:
            Cropped and resized face (H, W, 3)
        """
        h, w = image.shape[:2]
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, box)

        # Calculate face dimensions
        face_w = x2 - x1
        face_h = y2 - y1

        # Add margin
        margin_w = int(face_w * self.margin_ratio)
        margin_h = int(face_h * self.margin_ratio)

        # Expand box with margin (ensure integers)
        x1_margin = max(0, x1 - margin_w)
        y1_margin = max(0, y1 - margin_h)
        x2_margin = min(w, x2 + margin_w)
        y2_margin = min(h, y2 + margin_h)

        # Crop
        face_crop = image[y1_margin:y2_margin, x1_margin:x2_margin]

        # Resize to target size
        face_resized = cv2.resize(
            face_crop,
            (target_size[1], target_size[0]),  # (width, height)
            interpolation=cv2.INTER_LINEAR,
        )

        return face_resized

    def detect_and_crop(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Optional[np.ndarray]:
        """Detect face and crop in one step (convenience method).

        Args:
            image: RGB image
            target_size: Output size

        Returns:
            Cropped face if detected, None or fallback based on strategy

        Raises:
            FaceNotDetectedError: If no face detected and fallback_strategy is "error"
        """
        # Detect faces
        detections = self.detect_faces(image)

        # If face detected, crop and return
        if len(detections) > 0:
            # Use detection with highest confidence
            best_detection = max(detections, key=lambda d: d["confidence"])
            return self.crop_face(image, best_detection["box"], target_size)

        # No face detected - apply fallback strategy
        if self.fallback_strategy == "center_crop":
            # Return center crop as fallback
            return self._center_crop(image, target_size)

        elif self.fallback_strategy == "skip":
            # Return None to skip this image
            return None

        elif self.fallback_strategy == "error":
            # Raise error
            raise FaceNotDetectedError("No face detected in image")

        else:
            # Default to center crop
            return self._center_crop(image, target_size)

    def _center_crop(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Apply center crop as fallback.

        Args:
            image: Input image
            target_size: Output size (height, width)

        Returns:
            Center-cropped and resized image
        """
        h, w = image.shape[:2]
        crop_h, crop_w = target_size

        # Calculate crop size (square crop from center)
        crop_size = min(h, w)
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        end_h = start_h + crop_size
        end_w = start_w + crop_size

        # Crop
        cropped = image[start_h:end_h, start_w:end_w]

        # Resize
        resized = cv2.resize(
            cropped,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized

    def batch_detect_and_crop(
        self,
        images: List[np.ndarray],
        target_size: Tuple[int, int] = (224, 224),
    ) -> List[Optional[np.ndarray]]:
        """Process batch of images.

        Args:
            images: List of RGB images
            target_size: Output size

        Returns:
            List of cropped faces (None for failed detections if skip strategy)
        """
        results = []

        for image in images:
            try:
                face = self.detect_and_crop(image, target_size)
                results.append(face)
            except FaceNotDetectedError:
                results.append(None)

        return results
