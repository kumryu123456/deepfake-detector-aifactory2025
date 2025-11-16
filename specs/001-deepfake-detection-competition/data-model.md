# Data Model: Deepfake Detection Competition Model

**Feature**: Deepfake Detection AI Competition Platform
**Branch**: `001-deepfake-detection-competition`
**Created**: 2025-11-17

## 1. Overview

This document defines the data structures, model architecture, and data flow for the deepfake detection model implementation. The model processes both image and video inputs to produce binary classification predictions (Real=0, Fake=1) optimized for Macro F1-score.

## 2. Input Data Schema

### 2.1 Test Data Structure

**Directory Layout**:
```
./data/
├── image001.jpg
├── image002.png
├── video001.mp4
├── video002.mp4
└── ... (mixed images and videos, no subdirectories)
```

**File Specifications**:

| Property | Images | Videos |
|----------|--------|--------|
| **Formats** | JPG, PNG | MP4 |
| **Face Count** | Exactly 1 identifiable face | Exactly 1 identifiable face per frame |
| **Duration** | N/A | Average 5 seconds (~150 frames at 30fps) |
| **Audio** | N/A | No audio track |
| **Content** | Diverse ethnicities, ages | Diverse ethnicities, ages, motion |
| **Manipulation Types** | Face swap, lip sync, GAN-generated, legacy techniques | Face swap, lip sync, GAN-generated, legacy techniques |

**Data Characteristics**:
- Mixed real and fake samples
- Diverse generation sources: commercial services, open-source models, legacy tools
- May include compression artifacts
- Unknown real/fake ratio (must handle imbalance)

### 2.2 Output Data Schema

**File**: `submission.csv`

**Location**: Current working directory (same level as task.ipynb)

**Format**:
```csv
filename,label
image001.jpg,1
video001.mp4,0
image002.png,1
...
```

**Column Specifications**:

| Column | Type | Values | Requirements |
|--------|------|--------|--------------|
| `filename` | string | File name with extension | Must match input filename exactly (case-sensitive) |
| `label` | integer | 0 (Real) or 1 (Fake) | No null, None, or non-binary values allowed |

**Constraints**:
- One row per input file
- Order does not matter
- Must include ALL test files
- Video files get single prediction (not per-frame)

## 3. Model Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input Processing                        │
├─────────────────────────────────────────────────────────────────┤
│  Image Input (.jpg/.png)          Video Input (.mp4)           │
│          │                                  │                   │
│          ▼                                  ▼                   │
│   Face Detection              Frame Extraction (16-32 frames)   │
│   & Alignment                          │                        │
│          │                              ▼                        │
│          │                       Face Detection per Frame       │
│          │                              │                        │
│          └──────────────┬───────────────┘                       │
│                         ▼                                        │
│              Preprocessed Face Images                           │
│                   (224×224 or 384×384)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Dual-Branch Feature Extraction             │
├──────────────────────────────┬──────────────────────────────────┤
│   Spatial Branch             │    Frequency Branch              │
│                              │                                  │
│   EfficientNet-B4            │    FFT/DCT Transform             │
│   (Pretrained on ImageNet)   │    ┌──────────┬──────────┐     │
│          │                   │    │ Amplitude│  Phase   │     │
│          ▼                   │    │ Spectrum │ Spectrum │     │
│   Vision Transformer         │    └─────┬────┴────┬─────┘     │
│   Encoder (4 layers)         │          ▼         ▼           │
│          │                   │    Conv Layers in Freq Domain   │
│          ▼                   │          │                      │
│   512-dim feature vector     │          ▼                      │
│                              │    512-dim feature vector        │
└──────────────┬───────────────┴──────────────┬───────────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Fusion & Classification                    │
├─────────────────────────────────────────────────────────────────┤
│   Concatenate: [spatial_features | frequency_features]          │
│   1024-dim combined features                                     │
│          │                                                       │
│          ▼                                                       │
│   Self-Attention Layer (learn feature importance)               │
│          │                                                       │
│          ▼                                                       │
│   FC Layer: 1024 → 512 → 2                                      │
│          │                                                       │
│          ▼                                                       │
│   Softmax → [P(Real), P(Fake)]                                  │
│          │                                                       │
│          ▼                                                       │
│   Binary Prediction: argmax → {0, 1}                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Specifications

#### 3.2.1 Spatial Branch

**Backbone: EfficientNet-B4**

```python
# Model configuration
{
  "architecture": "efficientnet_b4",
  "input_size": (3, 224, 224),  # RGB channels, height, width
  "pretrained": "imagenet",
  "num_features": 1792,  # EfficientNet-B4 output features
  "dropout": 0.3
}
```

**Vision Transformer Encoder**

```python
# Transformer configuration
{
  "num_layers": 4,
  "num_heads": 8,
  "hidden_dim": 512,
  "mlp_dim": 2048,
  "dropout": 0.1,
  "input_dim": 1792,  # from EfficientNet
  "output_dim": 512
}
```

**Feature Extraction Flow**:
1. Input: RGB image (224×224×3)
2. EfficientNet-B4 forward pass → Feature map of shape (batch, 1792, 7, 7)
   - **Note**: EfficientNet-B4's final conv layer outputs 1792 channels with 7×7 spatial dimensions
   - This is BEFORE global average pooling (we bypass pooling for Transformer)
3. Reshape feature map to patch embeddings: (batch, 49, 1792)
   - **Clarification**: 49 patches = 7×7 spatial locations, each with 1792 features
   - Each spatial location is treated as a "patch" for the Transformer
4. Add learnable positional encoding: (batch, 49, 1792)
5. Transformer encoder processes sequence of 49 patches → (batch, 49, 512)
6. Global average pooling across patches → (batch, 512) final spatial feature vector

**Implementation Note**: Extract features from EfficientNet-B4 BEFORE the global pooling layer to preserve spatial information for Transformer processing.

#### 3.2.2 Frequency Branch

**FFT Configuration**

```python
# Frequency domain processing
{
  "method": "fft2d",  # 2D Fast Fourier Transform
  "components": ["amplitude", "phase"],
  "input_size": (224, 224),  # Applied to preprocessed face images (AFTER resizing to 224×224)
  "freq_filters": [
    {"type": "high_pass", "cutoff": 0.3},  # Capture high-freq artifacts
    {"type": "band_pass", "low": 0.1, "high": 0.7}
  ]
}
```

**Clarification - Frequency Branch Input**:
- FFT is applied to the **preprocessed RGB face image** of size 224×224
- This is the SAME input image that goes to the Spatial Branch
- The image is converted from spatial domain (224×224×3) to frequency domain
- For each RGB channel, we compute: amplitude = abs(FFT2D(channel)), phase = angle(FFT2D(channel))
- Result: amplitude (224×224×3) and phase (224×224×3), stacked to create (224×224×6) input for frequency CNN
- **Note**: Some implementations average across RGB channels first → (224×224) single-channel → (224×224×2) for amp+phase

**Frequency CNN**

```python
# Convolutional layers in frequency domain
{
  "layers": [
    {"type": "conv2d", "in_channels": 2, "out_channels": 64, "kernel": 3},  # amp + phase
    {"type": "relu"},
    {"type": "conv2d", "in_channels": 64, "out_channels": 128, "kernel": 3},
    {"type": "relu"},
    {"type": "adaptive_avg_pool2d", "output_size": (7, 7)},
    {"type": "flatten"},
    {"type": "fc", "in_features": 128*7*7, "out_features": 512}
  ]
}
```

#### 3.2.3 Fusion Layer

**Self-Attention Fusion**

```python
# Attention-based feature fusion
{
  "input_dim": 1024,  # 512 spatial + 512 frequency
  "attention_heads": 4,
  "attention_dim": 256,
  "output_dim": 512,
  "fusion_strategy": "hierarchical"  # or "late_fusion" for simpler approach
}
```

**Fusion Architecture Details**:

The fusion layer combines 1024-dim input (spatial 512 + frequency 512) into 512-dim output. This dimensionality reduction is **intentional and beneficial**:

1. **Information Redundancy**: Spatial and frequency features capture overlapping information about the same input image
2. **Attention Mechanism**: Self-attention learns to weight and select the most discriminative features from both branches
3. **Compression is Feature Selection**: Not all 1024 dimensions are equally important; attention compresses by selecting salient information

**Implementation Options**:

**Option A: Late Fusion (Simpler, Faster)**
```
Concatenate [spatial_512 | frequency_512] → 1024-dim
  ↓
Linear projection: 1024 → 512
  ↓
Output: 512-dim fused features
```
- **Information Loss**: Minimal if features are redundant
- **Speed**: Fast, single linear layer
- **Use when**: Speed is critical or features are highly correlated

**Option B: Hierarchical Fusion (Better Performance, Recommended)**
```
Step 1: Cross-Modal Attention
  Q = Linear(spatial_512),  K = Linear(frequency_512),  V = Linear(frequency_512)
  attended_freq = MultiHeadAttention(Q, K, V) → 512-dim

Step 2: Combine
  combined = Concatenate([spatial_512 | attended_freq_512]) → 1024-dim

Step 3: Self-Attention + Projection
  fused = SelfAttention(combined) → 1024-dim
  output = Linear(fused) → 512-dim
```
- **Information Loss**: Minimized through explicit cross-modal attention
- **Benefit**: Learns which frequency features complement spatial features
- **Trade-off**: +10-15ms inference time, +2-3% F1 improvement

**Verification**: To ensure minimal information loss, monitor validation performance:
- If fusion improves F1 over single-branch models → fusion is effective
- If F1 drops after fusion → indicates poor feature integration (debug attention weights)

**Classification Head**

```python
# Final classification layers
{
  "layers": [
    {"type": "fc", "in": 512, "out": 256, "dropout": 0.5},
    {"type": "relu"},
    {"type": "fc", "in": 256, "out": 2}  # Binary classification
  ],
  "activation": "softmax"
}
```

### 3.3 Video Processing Module

**Frame Sampling Strategy**

```python
# Video frame extraction configuration
{
  "target_frames": 16,  # Balance accuracy and speed
  "sampling_method": "uniform",  # Evenly spaced across video
  "fallback_fps": 30,
  "min_frames": 8,
  "max_frames": 32
}
```

**Temporal Aggregation**

```python
# Aggregate frame-level predictions to video-level
{
  "aggregation_method": "average_logits",  # Average before softmax
  "alternatives": [
    "max_confidence",  # Take most confident prediction
    "majority_vote"    # Most common prediction
  ],
  "confidence_threshold": 0.7  # For early stopping (optional)
}
```

**Processing Pipeline**:
1. Extract 16 frames uniformly from 5-second video
2. Detect face in each frame (skip frames without face)
3. Pass each frame through dual-branch model → 16 logit pairs
4. Average logits: `avg_logit = mean(logits, axis=0)`
5. Apply softmax: `probs = softmax(avg_logit)`
6. Predict: `label = argmax(probs)`

## 4. Training Data Structures

### 4.1 Training Dataset Schema

**Dataset Configuration**

```python
{
  "datasets": [
    {
      "name": "FaceForensics++",
      "path": "./data/faceforensics",
      "real_count": 1000,
      "fake_count": 4000,
      "compression": ["c0", "c23", "c40"],  # Raw, light, heavy
      "methods": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
      "weight": 0.3
    },
    {
      "name": "DFDC",
      "path": "./data/dfdc",
      "real_count": 23654,
      "fake_count": 104500,
      "weight": 0.5
    },
    {
      "name": "Celeb-DF",
      "path": "./data/celebdf",
      "real_count": 590,
      "fake_count": 5639,
      "weight": 0.2
    }
  ],
  "sampling_strategy": "balanced",  # Equal Real/Fake per batch
  "train_split": 0.8,
  "val_split": 0.1,
  "test_split": 0.1
}
```

### 4.2 Batch Structure

**Training Batch**

```python
{
  "batch_size": 32,
  "structure": {
    "images": Tensor(32, 3, 224, 224),  # RGB images
    "labels": Tensor(32),  # 0 or 1
    "dataset_ids": Tensor(32),  # Which dataset each sample came from
    "manipulation_types": List[str],  # Optional: track forgery method
    "metadata": {
      "original_sizes": List[Tuple[int, int]],
      "face_boxes": Tensor(32, 4),  # Bounding boxes
      "landmarks": Tensor(32, 68, 2)  # Facial landmarks (optional)
    }
  }
}
```

## 5. Preprocessing Pipeline

### 5.1 Face Detection Configuration

```python
{
  "detector": "retinaface",  # Recommended for robustness
  "confidence_threshold": 0.9,
  "margin_ratio": 0.3,  # Add 30% margin around face box
  "min_face_size": 80,  # Minimum face dimension in pixels
  "fallback_detector": "mtcnn"  # If RetinaFace fails
}
```

### 5.2 Preprocessing Transforms

**Training Transforms**

```python
{
  "resize": (256, 256),
  "augmentation": [
    {"name": "RandomHorizontalFlip", "p": 0.5},
    {"name": "RandomRotation", "degrees": 15},
    {"name": "ColorJitter", "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
    {"name": "GaussianBlur", "kernel_size": 5, "p": 0.05},
    {"name": "GaussianNoise", "sigma": 0.05, "p": 0.1},
    {"name": "JPEGCompression", "quality_range": (60, 100), "p": 0.5},
    {"name": "RandomCrop", "size": (224, 224)},
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
  ]
}
```

**Inference Transforms**

```python
{
  "resize": (224, 224),
  "transforms": [
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
  ]
}
```

### 5.3 Data Augmentation Pipeline

```python
# Using albumentations library
import albumentations as A

transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0)
    ], p=0.1),
    A.OneOf([
        A.OpticalDistortion(p=1.0),
        A.GridDistortion(p=1.0),
        A.ElasticTransform(p=1.0)
    ], p=0.1),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
    A.RandomCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 6. Model State & Checkpoints

### 6.1 Checkpoint Structure

```python
{
  "model_state_dict": OrderedDict,  # Model weights
  "optimizer_state_dict": OrderedDict,  # Optimizer state
  "epoch": int,
  "best_val_f1": float,
  "train_loss": float,
  "val_loss": float,
  "metrics": {
    "val_accuracy": float,
    "val_macro_f1": float,
    "val_precision_real": float,
    "val_precision_fake": float,
    "val_recall_real": float,
    "val_recall_fake": float
  },
  "config": {
    "model_architecture": str,
    "input_size": Tuple[int, int],
    "num_classes": int,
    "backbone": str
  },
  "timestamp": str
}
```

### 6.2 Model Versioning

```python
# Checkpoint naming convention
{
  "format": "model_{backbone}_{dataset}_{metric}_{value}_epoch{num}.pth",
  "examples": [
    "model_effb4_multi_f1_0.8523_epoch87.pth",
    "model_effb4hybrid_ffpp_f1_0.9102_epoch45.pth"
  ]
}
```

## 7. Inference Data Flow

### 7.1 Inference Pipeline

```
1. Load test data list from ./data/
   ├─ Scan directory for .jpg, .png, .mp4 files
   └─ Store filenames in list

2. Initialize model
   ├─ Load checkpoint weights
   ├─ Set model to eval mode
   └─ Move model to GPU

3. Process each file
   ├─ If image:
   │  ├─ Load image
   │  ├─ Detect face
   │  ├─ Crop & resize to 224×224
   │  ├─ Normalize
   │  ├─ Forward pass through model
   │  └─ Get prediction
   │
   └─ If video:
      ├─ Extract 16 frames uniformly
      ├─ For each frame:
      │  ├─ Detect face
      │  ├─ Crop & resize to 224×224
      │  └─ Normalize
      ├─ Batch forward pass through model
      ├─ Aggregate logits (average)
      └─ Get prediction

4. Store results
   ├─ Append (filename, label) to results list
   └─ Continue for all files

5. Save submission.csv
   ├─ Create DataFrame with columns: filename, label
   ├─ Write to ./submission.csv
   └─ Verify format (no null values, all files included)
```

### 7.2 Batch Inference Optimization

```python
{
  "batch_size": 64,  # Process multiple images in parallel
  "video_batch_size": 16,  # Process video frames in batch
  "num_workers": 8,  # DataLoader workers
  "pin_memory": True,
  "prefetch_factor": 2,
  "use_fp16": True,  # Mixed precision for 2x speedup
  "compile_model": False  # PyTorch 2.0 compile (optional)
}
```

## 8. Metrics & Evaluation Schema

### 8.1 Evaluation Metrics

```python
{
  "primary_metric": "macro_f1",
  "secondary_metrics": [
    "accuracy",
    "precision_real",
    "precision_fake",
    "recall_real",
    "recall_fake",
    "f1_real",
    "f1_fake",
    "auc_roc",
    "confusion_matrix"
  ]
}
```

### 8.2 Macro F1 Calculation

```python
# Macro F1 computation
def compute_macro_f1(y_true, y_pred):
    """
    y_true: ground truth labels (0 or 1)
    y_pred: predicted labels (0 or 1)
    """
    # Compute per-class F1
    f1_real = compute_f1(y_true, y_pred, positive_class=0)
    f1_fake = compute_f1(y_true, y_pred, positive_class=1)

    # Macro average
    macro_f1 = (f1_real + f1_fake) / 2

    return macro_f1

def compute_f1(y_true, y_pred, positive_class):
    """F1 for specific class"""
    # Binarize for this class
    y_true_bin = (y_true == positive_class).astype(int)
    y_pred_bin = (y_pred == positive_class).astype(int)

    # True Positives, False Positives, False Negatives
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()

    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1
```

## 9. Configuration Management

### 9.1 Master Configuration File

```yaml
# config.yaml
model:
  architecture: "dual_branch_hybrid"
  spatial_branch:
    backbone: "efficientnet_b4"
    pretrained: true
    freeze_backbone: false
    transformer_layers: 4
    transformer_heads: 8
    output_dim: 512
  frequency_branch:
    method: "fft2d"
    conv_channels: [64, 128, 256]
    output_dim: 512
  fusion:
    attention_heads: 4
    hidden_dim: 512
  classifier:
    hidden_dims: [256]
    dropout: 0.5
    num_classes: 2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer: "adamw"
  weight_decay: 0.01
  scheduler: "cosine_annealing"
  warmup_epochs: 5
  loss_function: "combined"
  loss_weights:
    cross_entropy: 0.5
    focal: 0.3
    f1: 0.2

data:
  input_size: [224, 224]
  datasets:
    - name: "faceforensics"
      path: "./data/faceforensics"
      weight: 0.3
    - name: "dfdc"
      path: "./data/dfdc"
      weight: 0.5
    - name: "celebdf"
      path: "./data/celebdf"
      weight: 0.2
  augmentation: "heavy"
  num_workers: 8

inference:
  batch_size: 64
  video_frames: 16
  video_sampling: "uniform"
  aggregation: "average_logits"
  use_fp16: true

evaluation:
  primary_metric: "macro_f1"
  cross_dataset_validation: true
  k_folds: 5
```

## 10. Summary

This data model defines:

1. **Input/Output Formats**: CSV schema for submission, test data structure
2. **Model Architecture**: Dual-branch (spatial + frequency) hybrid network
3. **Processing Pipelines**: Face detection, preprocessing, augmentation
4. **Training Configuration**: Multi-dataset training, batch structure
5. **Inference Flow**: Image and video processing, batch optimization
6. **Evaluation Schema**: Macro F1 calculation, metrics tracking

The architecture is designed to:
- Maximize Macro F1-score through balanced training and appropriate loss functions
- Ensure generalization through frequency domain analysis and multi-dataset training
- Meet inference time constraints through efficient batch processing
- Comply with all competition rules (single model, no ensembles, reproducible)

**Next Steps**: Create contracts for model interfaces and quickstart guide for implementation.
