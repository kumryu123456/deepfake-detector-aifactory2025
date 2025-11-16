# Model Interface Contract

**Feature**: Deepfake Detection AI Competition Platform
**Branch**: `001-deepfake-detection-competition`
**Created**: 2025-11-17

## 1. Overview

This document defines the programmatic interfaces for the deepfake detection model components. All components must adhere to these contracts to ensure modularity, testability, and maintainability.

## 2. Core Model Interface

### 2.1 DeepfakeDetector (Main Model)

**Purpose**: Primary interface for the complete deepfake detection model.

```python
class DeepfakeDetector(nn.Module):
    """
    Dual-branch hybrid model for deepfake detection.
    Combines spatial features (EfficientNet + Transformer) with
    frequency domain features (FFT/DCT analysis).
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the deepfake detector.

        Args:
            config: Model configuration object containing:
                - spatial_branch_config: Configuration for spatial branch
                - frequency_branch_config: Configuration for frequency branch
                - fusion_config: Configuration for feature fusion
                - num_classes: Number of output classes (default: 2)
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
               Normalized RGB images

        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
                   Raw logits before softmax
        """
        pass

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            predictions: Predicted class labels (0 or 1), shape (batch_size,)
            probabilities: Class probabilities, shape (batch_size, num_classes)
        """
        pass

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for analysis.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            features: Dictionary containing:
                - 'spatial_features': Tensor of shape (batch_size, 512)
                - 'frequency_features': Tensor of shape (batch_size, 512)
                - 'fused_features': Tensor of shape (batch_size, 1024)
        """
        pass
```

### 2.2 SpatialBranch

**Purpose**: Extract spatial features using CNN backbone + Transformer.

```python
class SpatialBranch(nn.Module):
    """
    Spatial feature extraction branch.
    Uses EfficientNet backbone followed by Vision Transformer encoder.
    """

    def __init__(self,
                 backbone: str = 'efficientnet_b4',
                 pretrained: bool = True,
                 transformer_layers: int = 4,
                 transformer_heads: int = 8,
                 output_dim: int = 512):
        """
        Initialize spatial branch.

        Args:
            backbone: Name of backbone architecture
            pretrained: Whether to use ImageNet pretrained weights
            transformer_layers: Number of transformer encoder layers
            transformer_heads: Number of attention heads
            output_dim: Output feature dimension
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features.

        Args:
            x: RGB images, shape (batch_size, 3, H, W)

        Returns:
            features: Spatial features, shape (batch_size, output_dim)
        """
        pass
```

### 2.3 FrequencyBranch

**Purpose**: Extract frequency domain features using FFT/DCT.

```python
class FrequencyBranch(nn.Module):
    """
    Frequency domain feature extraction branch.
    Applies FFT/DCT and processes amplitude/phase spectra.
    """

    def __init__(self,
                 method: str = 'fft',  # 'fft' or 'dct'
                 conv_channels: List[int] = [64, 128, 256],
                 output_dim: int = 512):
        """
        Initialize frequency branch.

        Args:
            method: Frequency transform method ('fft' or 'dct')
            conv_channels: List of channel dimensions for conv layers
            output_dim: Output feature dimension
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency domain features.

        Args:
            x: RGB images, shape (batch_size, 3, H, W)

        Returns:
            features: Frequency features, shape (batch_size, output_dim)
        """
        pass

    def compute_frequency_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute frequency domain representation.

        Args:
            x: Input images, shape (batch_size, 3, H, W)

        Returns:
            amplitude: Amplitude spectrum, shape (batch_size, H, W)
            phase: Phase spectrum, shape (batch_size, H, W)
        """
        pass
```

### 2.4 FusionLayer

**Purpose**: Combine spatial and frequency features with attention mechanism.

```python
class FusionLayer(nn.Module):
    """
    Feature fusion layer with self-attention.
    Combines spatial and frequency features adaptively.
    """

    def __init__(self,
                 input_dim: int = 1024,  # 512 + 512
                 attention_heads: int = 4,
                 hidden_dim: int = 512):
        """
        Initialize fusion layer.

        Args:
            input_dim: Combined feature dimension (spatial + frequency)
            attention_heads: Number of attention heads
            hidden_dim: Hidden dimension for attention
        """
        pass

    def forward(self,
                spatial_features: torch.Tensor,
                frequency_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse spatial and frequency features.

        Args:
            spatial_features: Shape (batch_size, 512)
            frequency_features: Shape (batch_size, 512)

        Returns:
            fused_features: Shape (batch_size, input_dim)
        """
        pass
```

## 3. Data Processing Interfaces

### 3.1 FaceDetector

**Purpose**: Detect and crop faces from images/video frames.

```python
class FaceDetector:
    """
    Face detection and cropping interface.
    Supports multiple detection backends (RetinaFace, MTCNN).
    """

    def __init__(self,
                 detector_name: str = 'retinaface',
                 confidence_threshold: float = 0.9,
                 margin_ratio: float = 0.3,
                 device: str = 'cuda'):
        """
        Initialize face detector.

        Args:
            detector_name: Detector backend ('retinaface', 'mtcnn', 'mediapipe')
            confidence_threshold: Minimum confidence for detection
            margin_ratio: Margin ratio around detected face box
            device: Device for computation ('cuda' or 'cpu')
        """
        pass

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all faces in an image.

        Args:
            image: RGB image as numpy array, shape (H, W, 3)

        Returns:
            detections: List of detection dictionaries, each containing:
                - 'box': [x1, y1, x2, y2] coordinates
                - 'confidence': Detection confidence score
                - 'landmarks': Optional facial landmarks (68 points)
        """
        pass

    def crop_face(self,
                  image: np.ndarray,
                  box: List[int],
                  target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Crop and resize face region.

        Args:
            image: RGB image as numpy array
            box: Face bounding box [x1, y1, x2, y2]
            target_size: Output size (height, width)

        Returns:
            face: Cropped and resized face, shape (target_size[0], target_size[1], 3)
        """
        pass

    def detect_and_crop(self,
                       image: np.ndarray,
                       target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Detect face and crop in one step (convenience method).

        Args:
            image: RGB image as numpy array
            target_size: Output size

        Returns:
            face: Cropped face if detected, None otherwise
        """
        pass
```

### 3.2 VideoProcessor

**Purpose**: Extract and process frames from video files.

```python
class VideoProcessor:
    """
    Video frame extraction and processing interface.
    """

    def __init__(self,
                 target_frames: int = 16,
                 sampling_method: str = 'uniform',
                 face_detector: Optional[FaceDetector] = None):
        """
        Initialize video processor.

        Args:
            target_frames: Number of frames to extract
            sampling_method: Sampling strategy ('uniform', 'random', 'adaptive')
            face_detector: Face detector instance for per-frame processing
        """
        pass

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video file.

        Args:
            video_path: Path to video file (.mp4)

        Returns:
            frames: List of RGB frames as numpy arrays, shape (H, W, 3)
        """
        pass

    def process_video(self,
                     video_path: str,
                     target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Extract frames and detect faces.

        Args:
            video_path: Path to video file
            target_size: Size for cropped faces

        Returns:
            faces: Tensor of face crops, shape (num_frames, 3, H, W)
                  num_frames may be < target_frames if some frames lack faces
        """
        pass
```

### 3.3 DataPreprocessor

**Purpose**: Apply preprocessing transforms to images.

```python
class DataPreprocessor:
    """
    Data preprocessing and augmentation interface.
    """

    def __init__(self,
                 mode: str = 'inference',  # 'train' or 'inference'
                 input_size: Tuple[int, int] = (224, 224),
                 augmentation_config: Optional[Dict] = None):
        """
        Initialize preprocessor.

        Args:
            mode: 'train' for augmentation, 'inference' for minimal preprocessing
            input_size: Target image size
            augmentation_config: Augmentation parameters (for training mode)
        """
        pass

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image.

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            tensor: Preprocessed image tensor, shape (3, H, W), normalized
        """
        pass

    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess batch of images.

        Args:
            images: List of RGB images as numpy arrays

        Returns:
            batch: Batched tensor, shape (batch_size, 3, H, W)
        """
        pass
```

## 4. Inference Interfaces

### 4.1 InferenceEngine

**Purpose**: High-level interface for running inference on test data.

```python
class InferenceEngine:
    """
    Main inference engine for competition submission.
    Processes test data directory and generates submission.csv.
    """

    def __init__(self,
                 model: DeepfakeDetector,
                 face_detector: FaceDetector,
                 video_processor: VideoProcessor,
                 preprocessor: DataPreprocessor,
                 device: str = 'cuda',
                 batch_size: int = 64,
                 use_fp16: bool = True):
        """
        Initialize inference engine.

        Args:
            model: Trained deepfake detector model
            face_detector: Face detection instance
            video_processor: Video processing instance
            preprocessor: Data preprocessing instance
            device: Computation device
            batch_size: Batch size for parallel processing
            use_fp16: Use mixed precision for speedup
        """
        pass

    def run_inference(self, data_dir: str, output_csv: str = 'submission.csv') -> pd.DataFrame:
        """
        Run inference on all test data and generate submission file.

        Args:
            data_dir: Path to test data directory (./data/)
            output_csv: Output CSV filename

        Returns:
            results: DataFrame with columns ['filename', 'label']

        Side Effects:
            Writes submission.csv to current directory
        """
        pass

    def process_image(self, image_path: str) -> int:
        """
        Process single image file.

        Args:
            image_path: Path to image file (.jpg or .png)

        Returns:
            label: Predicted label (0 for Real, 1 for Fake)
        """
        pass

    def process_video(self, video_path: str) -> int:
        """
        Process single video file.

        Args:
            video_path: Path to video file (.mp4)

        Returns:
            label: Predicted label (0 for Real, 1 for Fake)
        """
        pass

    def aggregate_frame_predictions(self,
                                   logits: torch.Tensor,
                                   method: str = 'average_logits') -> int:
        """
        Aggregate frame-level predictions to video-level.

        Args:
            logits: Frame-level logits, shape (num_frames, num_classes)
            method: Aggregation method ('average_logits', 'max_confidence', 'majority_vote')

        Returns:
            label: Video-level prediction (0 or 1)
        """
        pass
```

### 4.2 ModelLoader

**Purpose**: Load trained model weights and configuration.

```python
class ModelLoader:
    """
    Model checkpoint loading and initialization.
    """

    @staticmethod
    def load_checkpoint(checkpoint_path: str,
                       device: str = 'cuda',
                       eval_mode: bool = True) -> DeepfakeDetector:
        """
        Load model from checkpoint file.

        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to load model onto
            eval_mode: Set model to evaluation mode

        Returns:
            model: Loaded DeepfakeDetector instance
        """
        pass

    @staticmethod
    def load_config(config_path: str) -> ModelConfig:
        """
        Load model configuration from YAML file.

        Args:
            config_path: Path to config.yaml

        Returns:
            config: ModelConfig object
        """
        pass
```

## 5. Training Interfaces

### 5.1 Trainer

**Purpose**: Manage training loop and optimization.

```python
class Trainer:
    """
    Training orchestration interface.
    """

    def __init__(self,
                 model: DeepfakeDetector,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig,
                 device: str = 'cuda'):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Computation device
        """
        pass

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Execute training loop.

        Args:
            num_epochs: Number of training epochs

        Returns:
            history: Dictionary of training metrics over epochs
        """
        pass

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            metrics: Dictionary of training metrics for this epoch
        """
        pass

    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            metrics: Dictionary of validation metrics
        """
        pass

    def save_checkpoint(self, path: str, metrics: Dict[str, float]):
        """
        Save model checkpoint.

        Args:
            path: Checkpoint save path
            metrics: Current metrics to save with checkpoint
        """
        pass
```

### 5.2 LossFunction

**Purpose**: Combined loss for Macro F1 optimization.

```python
class CombinedLoss(nn.Module):
    """
    Combined loss function for Macro F1 optimization.
    Combines Cross-Entropy, Focal Loss, and differentiable F1 loss.
    """

    def __init__(self,
                 ce_weight: float = 0.5,
                 focal_weight: float = 0.3,
                 f1_weight: float = 0.2,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25):
        """
        Initialize combined loss.

        Args:
            ce_weight: Weight for cross-entropy loss
            focal_weight: Weight for focal loss
            f1_weight: Weight for F1 loss
            focal_gamma: Focal loss gamma parameter
            focal_alpha: Focal loss alpha parameter
        """
        pass

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            logits: Model outputs, shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)

        Returns:
            loss: Scalar combined loss
        """
        pass
```

## 6. Evaluation Interfaces

### 6.1 MetricsCalculator

**Purpose**: Compute evaluation metrics including Macro F1.

```python
class MetricsCalculator:
    """
    Evaluation metrics computation.
    """

    @staticmethod
    def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Macro F1-score (primary competition metric).

        Args:
            y_true: Ground truth labels, shape (n_samples,)
            y_pred: Predicted labels, shape (n_samples,)

        Returns:
            macro_f1: Macro-averaged F1-score
        """
        pass

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_probs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities (optional, for AUC)

        Returns:
            metrics: Dictionary containing:
                - 'macro_f1': Macro F1-score
                - 'accuracy': Overall accuracy
                - 'precision_real': Precision for Real class
                - 'precision_fake': Precision for Fake class
                - 'recall_real': Recall for Real class
                - 'recall_fake': Recall for Fake class
                - 'f1_real': F1 for Real class
                - 'f1_fake': F1 for Fake class
                - 'auc': AUC-ROC (if y_probs provided)
                - 'confusion_matrix': 2x2 confusion matrix
        """
        pass

    @staticmethod
    def print_metrics_report(metrics: Dict[str, float]):
        """
        Print formatted metrics report.

        Args:
            metrics: Dictionary of metrics from compute_all_metrics()
        """
        pass
```

## 7. Configuration Objects

### 7.1 ModelConfig

```python
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    spatial_branch_config: SpatialBranchConfig
    frequency_branch_config: FrequencyBranchConfig
    fusion_config: FusionConfig
    num_classes: int = 2
    dropout: float = 0.5
```

### 7.2 TrainingConfig

```python
@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = 'adamw'
    weight_decay: float = 0.01
    scheduler: str = 'cosine_annealing'
    warmup_epochs: int = 5
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'ce': 0.5, 'focal': 0.3, 'f1': 0.2
    })
```

### 7.3 InferenceConfig

```python
@dataclass
class InferenceConfig:
    """Inference configuration."""
    batch_size: int = 64
    video_frames: int = 16
    video_sampling: str = 'uniform'
    aggregation_method: str = 'average_logits'
    use_fp16: bool = True
    num_workers: int = 8
```

## 8. Error Handling

### 8.1 Custom Exceptions

```python
class DeepfakeDetectionError(Exception):
    """Base exception for deepfake detection errors."""
    pass

class FaceNotDetectedError(DeepfakeDetectionError):
    """Raised when no face is detected in an image/frame."""
    pass

class InvalidInputError(DeepfakeDetectionError):
    """Raised when input data format is invalid."""
    pass

class ModelLoadError(DeepfakeDetectionError):
    """Raised when model checkpoint loading fails."""
    pass

class InferenceError(DeepfakeDetectionError):
    """Raised when inference fails."""
    pass
```

## 9. Input Validation Requirements

All implementations MUST validate inputs according to these specifications:

### 9.1 Image Input Validation

```python
def validate_image_input(image: np.ndarray) -> None:
    """Validate raw image input before processing."""
    # Shape validation
    assert len(image.shape) == 3, f"Image must be 3D (H, W, C), got shape {image.shape}"
    assert image.shape[2] == 3, f"Image must have 3 channels (RGB), got {image.shape[2]}"

    # Dimension validation
    h, w, c = image.shape
    assert h >= 224 and w >= 224, f"Image dimensions must be at least 224x224, got {h}x{w}"

    # Data type validation
    assert image.dtype in [np.uint8, np.float32, np.float64], \
        f"Image dtype must be uint8 or float, got {image.dtype}"

    # Value range validation
    if image.dtype == np.uint8:
        assert image.min() >= 0 and image.max() <= 255, "uint8 image values out of range [0, 255]"
    elif image.dtype in [np.float32, np.float64]:
        assert image.min() >= 0.0 and image.max() <= 255.0, \
            "Float image values must be in range [0, 1] or [0, 255]"
```

### 9.2 Video Input Validation

```python
def validate_video_input(video_path: str) -> None:
    """Validate video file before processing."""
    # File existence
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    # Format validation
    assert video_path.lower().endswith('.mp4'), \
        f"Video must be .mp4 format, got {os.path.splitext(video_path)[1]}"

    # Readability check
    import cv2
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video file: {video_path}"

    # Frame count validation
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count >= 1, f"Video must contain at least 1 frame, got {frame_count}"

    cap.release()
```

### 9.3 Model Input Tensor Validation

```python
def validate_model_input(x: torch.Tensor) -> None:
    """Validate preprocessed tensor input to model."""
    # Shape validation
    assert len(x.shape) == 4, f"Input must be 4D (B, C, H, W), got shape {x.shape}"
    batch_size, channels, height, width = x.shape

    # Channel validation
    assert channels == 3, f"Input must have 3 channels (RGB), got {channels}"

    # Dimension validation
    assert height >= 224 and width >= 224, \
        f"Input dimensions must be at least 224x224, got {height}x{width}"

    # Data type validation
    assert x.dtype == torch.float32, f"Input dtype must be float32, got {x.dtype}"

    # Normalization check (warn if not normalized)
    if x.min() < -5.0 or x.max() > 5.0:
        import warnings
        warnings.warn(f"Input values may not be normalized: min={x.min():.2f}, max={x.max():.2f}")
```

### 9.4 Batch Size Validation

```python
def validate_batch_size(batch_size: int) -> None:
    """Validate batch size parameter."""
    assert isinstance(batch_size, int), f"Batch size must be int, got {type(batch_size)}"
    assert 1 <= batch_size <= 256, \
        f"Batch size must be in range [1, 256], got {batch_size}"
```

### 9.5 Device Consistency Validation

```python
def validate_device_consistency(tensors: List[torch.Tensor], expected_device: str) -> None:
    """Ensure all tensors are on the same device."""
    for i, tensor in enumerate(tensors):
        actual_device = str(tensor.device)
        assert expected_device in actual_device, \
            f"Tensor {i} on device {actual_device}, expected {expected_device}"
```

### 9.6 Error Handling for Invalid Inputs

All validation failures MUST raise `InvalidInputError` with descriptive messages:

```python
try:
    validate_image_input(image)
except AssertionError as e:
    raise InvalidInputError(f"Image validation failed: {str(e)}")
```

### 9.7 Validation Checklist

Before processing any input, implementations MUST verify:

- [ ] **Image Input**: Shape (H, W, 3), dtype uint8 or float32, H≥224, W≥224, values in valid range
- [ ] **Video Input**: Format .mp4, file exists, readable, contains ≥1 frame
- [ ] **Model Input**: Shape (B, 3, H, W), dtype float32, normalized to appropriate range
- [ ] **Batch Size**: Integer in range [1, 256]
- [ ] **Device Placement**: All tensors on consistent device (cuda/cpu)
- [ ] **File Paths**: Exist and are accessible before reading
- [ ] **Output Directory**: Writable before saving results

## 10. Contract Compliance

All implementations MUST:

1. **Follow Type Hints**: Use Python type annotations for all parameters and return values
2. **Handle Errors Gracefully**: Catch exceptions and provide meaningful error messages
3. **Document Thoroughly**: Include docstrings with Args, Returns, and Raises sections
4. **Validate Inputs**: Apply validation functions from Section 9 before processing
5. **Log Appropriately**: Use logging module for debugging and monitoring
6. **Be Reproducible**: Support setting random seeds for deterministic behavior

## 11. Testing Requirements

Each interface implementation MUST have:

1. **Unit Tests**: Test individual methods in isolation
2. **Integration Tests**: Test component interactions
3. **Contract Tests**: Verify adherence to interface specifications
4. **Performance Tests**: Ensure inference time meets 3-hour constraint

Example test structure:
```python
def test_deepfake_detector_forward():
    """Test forward pass with valid input."""
    model = DeepfakeDetector(config)
    x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    logits = model.forward(x)
    assert logits.shape == (4, 2)  # Correct output shape

def test_face_detector_with_no_face():
    """Test face detector error handling."""
    detector = FaceDetector()
    blank_image = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(FaceNotDetectedError):
        detector.detect_and_crop(blank_image)
```

## 12. Summary

This contract defines:

- **Core Model Interfaces**: DeepfakeDetector, SpatialBranch, FrequencyBranch, FusionLayer
- **Data Processing**: FaceDetector, VideoProcessor, DataPreprocessor
- **Inference**: InferenceEngine, ModelLoader
- **Training**: Trainer, LossFunction
- **Evaluation**: MetricsCalculator
- **Configuration**: ModelConfig, TrainingConfig, InferenceConfig
- **Error Handling**: Custom exception hierarchy

All implementations must comply with these interfaces to ensure system coherence and testability.
