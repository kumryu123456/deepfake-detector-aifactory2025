# Quickstart Guide: Deepfake Detection Competition

**Feature**: Deepfake Detection AI Competition Platform
**Branch**: `001-deepfake-detection-competition`
**Created**: 2025-11-17

## 1. Introduction

This quickstart guide provides step-by-step instructions to develop, train, and submit a competitive deepfake detection model for the National Forensic Service AI Competition.

**Competition Overview**:
- **Task**: Binary classification of face images/videos (Real=0, Fake=1)
- **Metric**: Macro F1-score
- **Constraint**: 3-hour inference time limit, single model only
- **Deadline**: November 20, 2025, 5 PM

## 2. Environment Setup

### 2.1 Choose CUDA Environment

The competition provides three CUDA environments. **Recommended: CUDA 11.8**

| Environment | Python | PyTorch | Use Case |
|-------------|--------|---------|----------|
| CUDA 10.2 | 3.8 | 1.6.0 | Legacy compatibility |
| **CUDA 11.8** | **3.9** | **1.13.1** | **Recommended (balanced)** |
| CUDA 12.6 | 3.10 | 2.7.1 | Latest features |

### 2.2 Local Development Setup

```bash
# Create virtual environment
conda create -n deepfake python=3.9
conda activate deepfake

# Install PyTorch (CUDA 11.8)
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install timm==0.9.12  # EfficientNet and ViT models
pip install opencv-python-headless==4.8.1.78
pip install albumentations==1.3.1  # Data augmentation
pip install pandas==2.1.4
pip install scikit-learn==1.3.2
pip install scipy==1.11.4
pip install numpy==1.24.3
pip install Pillow==10.1.0

# Install face detection (choose one)
pip install facenet-pytorch==2.5.3  # For MTCNN
# OR
pip install retinaface-pytorch==0.0.8  # For RetinaFace (recommended)

# Install utilities
pip install pyyaml==6.0.1
pip install tqdm==4.66.1
pip install matplotlib==3.8.2
pip install seaborn==0.13.0

# For submission
pip install aifactory  # Competition submission library
```

### 2.3 Verify Installation

```python
import torch
import torchvision
import timm
import cv2
import albumentations as A

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Test model creation
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
print(f"✓ EfficientNet-B4 loaded successfully")
```

## 3. Data Preparation

### 3.1 Download Training Datasets

**Recommended datasets** (use all for best generalization):

1. **FaceForensics++** (FF++):
   ```bash
   # Download from: https://github.com/ondyari/FaceForensics
   # Choose compression levels: c0 (raw), c23 (light), c40 (heavy)
   ```

2. **Deepfake Detection Challenge (DFDC)**:
   ```bash
   # Download from: https://ai.facebook.com/datasets/dfdc/
   # Large dataset: ~470GB
   ```

3. **Celeb-DF v2**:
   ```bash
   # Download from: https://github.com/yuezunli/celeb-deepfakeforensics
   ```

### 3.2 Organize Data Structure

```
data/
├── faceforensics/
│   ├── real/
│   │   ├── video001.mp4
│   │   └── ...
│   └── fake/
│       ├── video001.mp4
│       └── ...
├── dfdc/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── val/
├── celebdf/
│   ├── real/
│   └── fake/
└── sample/  # Competition sample data
    ├── fake_images/  # 7 samples
    └── fake_videos/  # 5 samples
```

### 3.3 Download Competition Sample Data

```python
# Download from competition page
# Place in ./data/sample/
# Use to verify your pipeline works correctly
```

## 4. Project Structure

```
deepfake-detection/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deepfake_detector.py     # Main model
│   │   ├── spatial_branch.py        # Spatial feature extraction
│   │   ├── frequency_branch.py      # Frequency feature extraction
│   │   └── fusion_layer.py          # Feature fusion
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # Dataset classes
│   │   ├── face_detector.py         # Face detection
│   │   ├── video_processor.py       # Video frame extraction
│   │   └── transforms.py            # Data augmentation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   ├── losses.py                # Loss functions
│   │   └── metrics.py               # Evaluation metrics
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── inference_engine.py      # Main inference
│   │   └── model_loader.py          # Load checkpoints
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       └── logger.py                # Logging utilities
├── configs/
│   ├── model_config.yaml            # Model architecture config
│   ├── training_config.yaml         # Training hyperparameters
│   └── inference_config.yaml        # Inference settings
├── scripts/
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation script
│   └── preprocess_data.py           # Data preprocessing
├── notebooks/
│   ├── eda.ipynb                    # Exploratory data analysis
│   └── task.ipynb                   # Competition submission notebook
├── checkpoints/                     # Model weights
├── logs/                            # Training logs
├── data/                            # Dataset directory
└── requirements.txt                 # Python dependencies
```

## 5. Implementation Phases

### Phase 1: Baseline Model (Week 1-2)

**Goal**: Create a working pipeline with basic EfficientNet model

**Steps**:

1. **Implement Face Detection**:
   ```python
   # src/data/face_detector.py
   from facenet_pytorch import MTCNN

   class FaceDetector:
       def __init__(self):
           self.detector = MTCNN(keep_all=False, device='cuda')

       def detect_and_crop(self, image):
           # Detect face and return cropped region
           pass
   ```

2. **Create Dataset Class**:
   ```python
   # src/data/dataset.py
   from torch.utils.data import Dataset

   class DeepfakeDataset(Dataset):
       def __init__(self, data_dir, transform=None):
           # Load image/video paths and labels
           pass

       def __getitem__(self, idx):
           # Return preprocessed image and label
           pass
   ```

3. **Build Baseline Model**:
   ```python
   # src/models/deepfake_detector.py
   import timm
   import torch.nn as nn

   class BaselineDetector(nn.Module):
       def __init__(self):
           super().__init__()
           self.backbone = timm.create_model('efficientnet_b4',
                                             pretrained=True,
                                             num_classes=2)

       def forward(self, x):
           return self.backbone(x)
   ```

4. **Train Baseline**:
   ```bash
   python scripts/train.py --config configs/baseline_config.yaml
   ```

5. **Validate Performance**:
   - Target: >85% accuracy on FaceForensics++ test set
   - Monitor Macro F1-score (not just accuracy)

### Phase 2: Hybrid Architecture (Week 3-4)

**Goal**: Add frequency branch and multi-dataset training

**Steps**:

1. **Implement Frequency Branch**:
   ```python
   # src/models/frequency_branch.py
   import torch
   import torch.fft

   class FrequencyBranch(nn.Module):
       def forward(self, x):
           # Apply FFT
           freq = torch.fft.fft2(x)
           amplitude = torch.abs(freq)
           phase = torch.angle(freq)

           # Process with CNN
           features = self.conv_layers(torch.stack([amplitude, phase], dim=1))
           return features
   ```

2. **Create Dual-Branch Model**:
   ```python
   # src/models/deepfake_detector.py
   class DualBranchDetector(nn.Module):
       def __init__(self):
           self.spatial_branch = SpatialBranch()
           self.frequency_branch = FrequencyBranch()
           self.fusion = FusionLayer()
           self.classifier = nn.Linear(1024, 2)

       def forward(self, x):
           spatial_feat = self.spatial_branch(x)
           freq_feat = self.frequency_branch(x)
           fused = self.fusion(spatial_feat, freq_feat)
           return self.classifier(fused)
   ```

3. **Multi-Dataset Training**:
   ```python
   # scripts/train.py
   datasets = [
       DeepfakeDataset('data/faceforensics', weight=0.3),
       DeepfakeDataset('data/dfdc', weight=0.5),
       DeepfakeDataset('data/celebdf', weight=0.2)
   ]
   train_loader = create_combined_loader(datasets)
   ```

4. **Validate Cross-Dataset**:
   - Train on FF++ + DFDC, validate on Celeb-DF
   - Target: >80% cross-dataset F1-score

### Phase 3: Optimization (Week 5)

**Goal**: Optimize for Macro F1-score and inference speed

**Steps**:

1. **Implement Macro F1 Loss**:
   ```python
   # src/training/losses.py
   class MacroF1Loss(nn.Module):
       def forward(self, logits, targets):
           # Differentiable F1 approximation
           pass

   class CombinedLoss(nn.Module):
       def __init__(self):
           self.ce_loss = nn.CrossEntropyLoss()
           self.focal_loss = FocalLoss()
           self.f1_loss = MacroF1Loss()

       def forward(self, logits, targets):
           return (0.5 * self.ce_loss(logits, targets) +
                   0.3 * self.focal_loss(logits, targets) +
                   0.2 * self.f1_loss(logits, targets))
   ```

2. **Optimize Video Processing**:
   ```python
   # src/data/video_processor.py
   class VideoProcessor:
       def extract_frames(self, video_path, num_frames=16):
           # Uniform sampling
           cap = cv2.VideoCapture(video_path)
           total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
           indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
           # Extract frames at indices
           pass
   ```

3. **Enable Mixed Precision**:
   ```python
   # scripts/train.py
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   for batch in train_loader:
       with autocast():
           outputs = model(batch['images'])
           loss = criterion(outputs, batch['labels'])
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

4. **Benchmark Inference Speed**:
   ```bash
   python scripts/benchmark_inference.py
   # Ensure inference completes in <2 hours for safety
   ```

### Phase 4: Submission (Week 6)

**Goal**: Create final submission notebook and verify reproducibility

**Steps**:

1. **Create task.ipynb**:
   ```python
   # notebooks/task.ipynb

   # Cell 1: Install dependencies
   !pip install torch torchvision timm opencv-python-headless facenet-pytorch

   # Cell 2: Load model
   import torch
   from models.deepfake_detector import DualBranchDetector

   model = DualBranchDetector()
   checkpoint = torch.load('./checkpoints/best_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   model.cuda()

   # Cell 3: Run inference
   from inference.inference_engine import InferenceEngine

   engine = InferenceEngine(model)
   results = engine.run_inference('./data/', 'submission.csv')

   print(f"Processed {len(results)} files")
   print(results.head())
   ```

2. **Test Locally**:
   ```bash
   # Simulate competition environment
   mkdir test_submission
   cp -r checkpoints test_submission/
   cp notebooks/task.ipynb test_submission/
   cd test_submission
   jupyter nbconvert --to notebook --execute task.ipynb
   # Verify submission.csv is created correctly
   ```

3. **Verify Reproducibility**:
   ```python
   # Run inference 3 times and verify identical results
   for i in range(3):
       results = engine.run_inference('./data/', f'submission_{i}.csv')
       # Compare submissions
   ```

4. **Submit to Competition**:
   ```python
   # Cell 4: Submit (in task.ipynb)
   import aifactory

   # Use CUDA 11.8 competition key
   competition_key = "YOUR_CUDA_11.8_KEY"
   aifactory.score.submit(competition_key)
   ```

## 6. Training Best Practices

### 6.1 Hyperparameters

**Recommended starting values**:

```yaml
# configs/training_config.yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adamw
  weight_decay: 0.01
  scheduler: cosine_annealing
  warmup_epochs: 5

loss:
  type: combined
  weights:
    cross_entropy: 0.5
    focal: 0.3
    f1: 0.2

augmentation:
  horizontal_flip: 0.5
  rotation: 15
  color_jitter: 0.5
  gaussian_blur: 0.05
  gaussian_noise: 0.1
  jpeg_compression: 0.5
```

### 6.2 Training Schedule

```python
# Warmup phase (epochs 1-5)
for epoch in range(1, 6):
    # Low learning rate, freeze backbone
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-5)
    train_epoch()

# Main training (epochs 6-90)
for epoch in range(6, 91):
    # Unfreeze all, use full LR with cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=85)
    train_epoch()
    scheduler.step()

# Fine-tuning (epochs 91-100)
for epoch in range(91, 101):
    # Very low LR, optimize for F1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    train_epoch()
```

### 6.3 Monitoring Metrics

**Track during training**:
- Train loss, validation loss
- **Macro F1-score** (primary metric)
- Per-class F1 (Real and Fake separately)
- Precision and Recall for both classes
- Accuracy (for reference only)

**Early stopping**: Stop if validation Macro F1 doesn't improve for 15 epochs

**Checkpoint strategy**: Save model when validation Macro F1 improves

## 7. Inference Pipeline

### 7.1 Inference Code Structure

```python
# src/inference/inference_engine.py

class InferenceEngine:
    def __init__(self, model, face_detector, video_processor):
        self.model = model.eval().cuda()
        self.face_detector = face_detector
        self.video_processor = video_processor

    def run_inference(self, data_dir, output_csv):
        # List all files
        files = list_files(data_dir)
        results = []

        # Process in batches
        for batch in batch_files(files, batch_size=64):
            images = []
            filenames = []

            for file_path in batch:
                if is_image(file_path):
                    img = process_image(file_path)
                elif is_video(file_path):
                    img = process_video(file_path)

                images.append(img)
                filenames.append(os.path.basename(file_path))

            # Batch inference
            with torch.no_grad():
                batch_tensor = torch.stack(images).cuda()
                logits = self.model(batch_tensor)
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

            for filename, pred in zip(filenames, predictions):
                results.append({'filename': filename, 'label': int(pred)})

        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        return df

    def process_video(self, video_path):
        # Extract frames
        frames = self.video_processor.extract_frames(video_path, num_frames=16)

        # Detect faces in each frame
        face_crops = []
        for frame in frames:
            face = self.face_detector.detect_and_crop(frame)
            if face is not None:
                face_crops.append(face)

        # Aggregate predictions
        if len(face_crops) == 0:
            # Fallback: use entire frame
            face_crops = [cv2.resize(frames[0], (224, 224))]

        # Process all frames
        face_tensors = [self.preprocess(f) for f in face_crops]
        batch = torch.stack(face_tensors).cuda()

        with torch.no_grad():
            logits = self.model(batch)
            # Average logits across frames
            avg_logits = torch.mean(logits, dim=0, keepdim=True)
            prediction = torch.argmax(avg_logits, dim=1).item()

        return prediction
```

### 7.2 Optimization Tricks

```python
# Use mixed precision
from torch.cuda.amp import autocast

with torch.no_grad():
    with autocast():
        logits = model(batch)

# Batch processing
def batch_iterator(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]

# Pre-allocate tensors
batch_tensor = torch.zeros(batch_size, 3, 224, 224, device='cuda')

# Parallel video frame extraction
from multiprocessing import Pool
with Pool(8) as p:
    frames = p.map(extract_frame, frame_indices)
```

## 8. Debugging & Troubleshooting

### 8.1 Common Issues

**Issue 1: Face not detected**
```python
# Solution: Use fallback to entire frame
try:
    face = face_detector.detect_and_crop(image)
except FaceNotDetectedError:
    # Use entire image with center crop
    face = center_crop_and_resize(image, target_size=(224, 224))
```

**Issue 2: Out of Memory (OOM)**
```python
# Solution: Reduce batch size
# In task.ipynb:
inference_config.batch_size = 32  # Instead of 64
inference_config.video_batch_size = 8  # Instead of 16

# For videos, reduce frames
video_processor.target_frames = 8  # Instead of 16
```

**Issue 3: Inference too slow**
```python
# Solutions:
# 1. Use FP16
torch.set_float32_matmul_precision('medium')

# 2. Reduce video frames
video_frames = 8  # Minimum

# 3. Use smaller model
model = timm.create_model('efficientnet_b3', ...)  # Instead of B4
```

**Issue 4: submission.csv format error**
```python
# Verify format
import pandas as pd

df = pd.read_csv('submission.csv')
assert list(df.columns) == ['filename', 'label']
assert df['label'].isin([0, 1]).all()
assert df['filename'].str.contains('\.(jpg|png|mp4)$', regex=True).all()
print("✓ submission.csv format is correct")
```

### 8.2 Local Testing

```python
# scripts/test_submission.py

def test_submission_format(csv_path, data_dir=None):
    """Validate submission.csv format with comprehensive checks."""
    df = pd.read_csv(csv_path)

    # Check filename is exactly "submission.csv" (case-sensitive)
    assert os.path.basename(csv_path) == 'submission.csv', "File must be named exactly 'submission.csv'"

    # Check columns
    assert list(df.columns) == ['filename', 'label'], "Wrong columns - must be ['filename', 'label']"

    # Check label values (integers only)
    assert df['label'].dtype in [int, 'int64'], "Labels must be integers"
    assert df['label'].isin([0, 1]).all(), "Labels must be 0 or 1 only"

    # Check for null/NaN/None values
    assert not df.isnull().any().any(), "No null/NaN values allowed"
    assert not (df['label'] == 'None').any(), "No 'None' string values allowed"
    assert not (df['filename'] == 'None').any(), "No 'None' string values allowed"

    # Check filenames have extensions (case-insensitive)
    assert df['filename'].str.fullmatch(r'.*\.(?:jpg|png|mp4)', case=False).all(), "All filenames must end with .jpg, .png, or .mp4 (case-insensitive)"

    # Check for duplicate filenames
    assert not df['filename'].duplicated().any(), "No duplicate filenames allowed"

    # If data directory provided, verify all input files are included
    if data_dir:
        input_files = set(os.listdir(data_dir))
        submission_files = set(df['filename'].values)
        missing = input_files - submission_files
        extra = submission_files - input_files
        assert len(missing) == 0, f"Missing predictions for files: {missing}"
        assert len(extra) == 0, f"Extra predictions for non-existent files: {extra}"

    print("✓ All validation checks passed")
    print(f"  - Total predictions: {len(df)}")
    print(f"  - Real (0): {(df['label'] == 0).sum()}")
    print(f"  - Fake (1): {(df['label'] == 1).sum()}")

def validate_and_fix_submission(csv_path, data_dir=None, auto_fix=False):
    """
    Enhanced validation with error detection and optional auto-fix.

    Args:
        csv_path: Path to submission.csv
        data_dir: Optional path to test data directory for completeness check
        auto_fix: If True, attempt to fix common errors automatically

    Returns:
        tuple: (is_valid: bool, errors: List[str], warnings: List[str])
    """
    errors = []
    warnings = []

    # Read CSV with error handling
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return False, [f"File not found: {csv_path}"], []
    except pd.errors.EmptyDataError:
        return False, ["CSV file is empty"], []
    except Exception as e:
        return False, [f"Failed to read CSV: {str(e)}"], []

    # Validate filename
    if os.path.basename(csv_path) != 'submission.csv':
        errors.append(f"File must be named 'submission.csv', got '{os.path.basename(csv_path)}'")

    # Validate columns
    if list(df.columns) != ['filename', 'label']:
        errors.append(f"Invalid columns: {list(df.columns)}. Expected: ['filename', 'label']")
        return False, errors, warnings

    # Check for empty DataFrame
    if len(df) == 0:
        errors.append("CSV contains no predictions")
        return False, errors, warnings

    # Validate label types and values
    if df['label'].dtype not in [int, 'int64', 'int32']:
        if auto_fix:
            try:
                df['label'] = df['label'].astype(int)
                warnings.append("Auto-fixed: Converted labels to integer type")
            except ValueError:
                errors.append(f"Labels must be integers. Found dtype: {df['label'].dtype}")
        else:
            errors.append(f"Labels must be integers. Found dtype: {df['label'].dtype}")

    # Check for invalid label values
    invalid_labels = df[~df['label'].isin([0, 1])]
    if len(invalid_labels) > 0:
        errors.append(f"Found {len(invalid_labels)} invalid label values (must be 0 or 1)")
        errors.append(f"  Invalid rows: {invalid_labels.index.tolist()[:10]}")  # Show first 10

    # Check for null/NaN values
    null_labels = df[df['label'].isnull()]
    if len(null_labels) > 0:
        errors.append(f"Found {len(null_labels)} null/NaN labels at rows: {null_labels.index.tolist()}")

    null_filenames = df[df['filename'].isnull()]
    if len(null_filenames) > 0:
        errors.append(f"Found {len(null_filenames)} null/NaN filenames at rows: {null_filenames.index.tolist()}")

    # Check for "None" string values
    none_labels = df[df['label'] == 'None']
    none_filenames = df[df['filename'] == 'None']
    if len(none_labels) > 0:
        errors.append(f"Found {len(none_labels)} string 'None' values in label column")
    if len(none_filenames) > 0:
        errors.append(f"Found {len(none_filenames)} string 'None' values in filename column")

    # Validate filename extensions (case-insensitive)
    invalid_extensions = df[~df['filename'].str.fullmatch(r'.*\.(?:jpg|png|mp4)', case=False)]
    if len(invalid_extensions) > 0:
        errors.append(f"Found {len(invalid_extensions)} files with invalid extensions")
        errors.append(f"  Examples: {invalid_extensions['filename'].head().tolist()}")

    # Check for duplicate filenames
    duplicates = df[df['filename'].duplicated(keep=False)]
    if len(duplicates) > 0:
        errors.append(f"Found {len(duplicates)} duplicate filenames")
        errors.append(f"  Duplicates: {duplicates['filename'].unique().tolist()}")

    # Validate against data directory if provided
    if data_dir:
        try:
            input_files = set(os.listdir(data_dir))
            submission_files = set(df['filename'].values)

            missing = input_files - submission_files
            extra = submission_files - input_files

            if len(missing) > 0:
                errors.append(f"Missing predictions for {len(missing)} files")
                errors.append(f"  First 10 missing: {list(missing)[:10]}")

            if len(extra) > 0:
                warnings.append(f"Found {len(extra)} predictions for non-existent files")
                warnings.append(f"  First 10 extra: {list(extra)[:10]}")
        except Exception as e:
            warnings.append(f"Could not validate against data directory: {str(e)}")

    # Summary
    is_valid = len(errors) == 0
    return is_valid, errors, warnings

# Example usage in inference pipeline
def create_submission_with_validation(predictions_dict, output_path='submission.csv', data_dir=None):
    """
    Create and validate submission CSV with error checking.

    Args:
        predictions_dict: Dict mapping filename -> label
        output_path: Path to save submission.csv
        data_dir: Optional path to validate against test data

    Returns:
        bool: True if submission created and validated successfully
    """
    import pandas as pd

    # Create DataFrame
    df = pd.DataFrame([
        {'filename': filename, 'label': int(label)}
        for filename, label in predictions_dict.items()
    ])

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} predictions")

    # Validate
    is_valid, errors, warnings = validate_and_fix_submission(output_path, data_dir)

    # Report results
    if warnings:
        print("\n⚠ Warnings:")
        for warning in warnings:
            print(f"  {warning}")

    if errors:
        print("\n❌ Validation FAILED:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\n✅ Validation PASSED")
        print(f"  - Total: {len(df)} predictions")
        print(f"  - Real (0): {(df['label'] == 0).sum()}")
        print(f"  - Fake (1): {(df['label'] == 1).sum()}")
        return True

def test_inference_time(model, data_dir):
    """Ensure inference completes within time limit."""
    import time

    start = time.time()
    engine = InferenceEngine(model)
    results = engine.run_inference(data_dir)
    elapsed = time.time() - start

    print(f"Inference time: {elapsed/60:.2f} minutes")
    assert elapsed < 10800, f"Inference too slow: {elapsed}s > 3 hours"
    print("✓ Inference time within limit")
```

## 9. Submission Checklist

Before final submission:

- [ ] Model trained on multiple datasets (FF++, DFDC, Celeb-DF)
- [ ] Validation Macro F1 > 80%
- [ ] Inference tested locally and completes in <2 hours
- [ ] submission.csv format validated
- [ ] task.ipynb runs end-to-end without errors
- [ ] All dependencies installed in notebook (pip install cells)
- [ ] Model checkpoint files included in submission directory
- [ ] No .git folders or unnecessary files
- [ ] Random seeds fixed for reproducibility
- [ ] Tested on competition sample data
- [ ] Competition key retrieved for CUDA 11.8
- [ ] Preprocessing steps documented in code comments

## 10. Expected Timeline

| Week | Phase | Tasks | Milestone |
|------|-------|-------|-----------|
| 1-2 | Baseline | Setup, data prep, baseline model | 85%+ accuracy on FF++ |
| 3-4 | Hybrid Model | Frequency branch, multi-dataset | 80%+ cross-dataset F1 |
| 5 | Optimization | F1 loss, video optimization | <2 hour inference |
| 6 | Submission | task.ipynb, testing, submit | Final submission |

## 11. Competition Keys

Retrieve your keys from: My Page > Activity History > Competition

| Environment | Competition Name |
|-------------|------------------|
| CUDA 11.8 | 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회 |
| CUDA 12.6 | 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회 (CUDA 12.6) |
| CUDA 10.2 | 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회 (CUDA 10.2) |

## 12. Resources

**Documentation**:
- Competition Q&A: https://aifactory.space/task/9197/qna
- Contact: cs@aifactory.page

**Code Resources**:
- DeepfakeBench: https://github.com/SCLBD/DeepfakeBench
- timm library: https://github.com/huggingface/pytorch-image-models
- Albumentations: https://albumentations.ai/

**Research Papers**:
- See research.md for comprehensive literature review

## 13. Quick Command Reference

```bash
# Setup environment
conda create -n deepfake python=3.9
conda activate deepfake
pip install -r requirements.txt

# Train baseline
python scripts/train.py --config configs/baseline_config.yaml

# Train hybrid model
python scripts/train.py --config configs/hybrid_config.yaml

# Evaluate model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data data/test/

# Run inference
python scripts/inference.py --checkpoint checkpoints/best_model.pth --data ./data/ --output submission.csv

# Test submission format
python scripts/test_submission.py submission.csv

# Submit to competition
# (Run in task.ipynb)
import aifactory
aifactory.score.submit("YOUR_COMPETITION_KEY")
```

## 14. Next Steps

After reading this quickstart:

1. **Review**: Read `research.md` for technical deep-dive
2. **Design**: Study `data-model.md` for architecture details
3. **Contracts**: Review `contracts/model-interface.md` for API specifications
4. **Implement**: Follow Phase 1 to start building your baseline
5. **Iterate**: Progress through phases 2-4
6. **Submit**: Create task.ipynb and submit before deadline

Good luck with the competition! 🚀
