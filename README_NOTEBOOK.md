# task.ipynb - Competition Submission Notebook

This notebook implements the complete inference pipeline for the **딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회** (Deepfake Crime Prevention AI Detection Model Competition).

## Overview

The `task.ipynb` notebook provides a complete end-to-end inference pipeline that:
- Loads a trained deepfake detection model
- Processes mixed image/video test data
- Generates competition-compliant submission.csv
- Validates the submission format
- Submits to AI Factory for automated scoring

## Requirements

### Environment
- **Python**: 3.8+
- **CUDA**: 11.8
- **PyTorch**: 1.13.1+cu118
- **GPU**: Recommended (NVIDIA GPU with at least 8GB VRAM)

### Prerequisites
1. Trained model checkpoint saved at `checkpoints/best.pth`
2. Test data available in `./data/` directory
3. All source code in `./src/` directory

## Quick Start

### 1. Prepare Environment

Ensure you have the following directory structure:
```
.
├── task.ipynb              # Competition submission notebook
├── checkpoints/
│   └── best.pth           # Trained model checkpoint
├── data/                  # Test data directory
│   ├── image1.jpg
│   ├── video1.mp4
│   └── ...
└── src/                   # Source code
    ├── inference/
    ├── models/
    ├── data/
    └── ...
```

### 2. Run the Notebook

#### Option A: On AI Factory Platform
1. Upload `task.ipynb` to the AI Factory platform
2. Ensure `checkpoints/best.pth` and `./src/` are uploaded
3. Run all cells sequentially
4. The final cell will automatically submit to AI Factory

#### Option B: Local Testing
1. Open `task.ipynb` in Jupyter Lab/Notebook
2. Run all cells sequentially
3. Check that `submission.csv` is generated
4. Verify format with validation cell
5. The AI Factory submission cell will be skipped locally

### 3. Expected Output

After running all cells, you should see:
```
submission.csv              # Competition submission file
```

The `submission.csv` format:
```csv
filename,label
image1.jpg,0
video1.mp4,1
...
```

Where:
- `filename`: Name of the test file
- `label`: Prediction (0 = Real, 1 = Fake)

## Notebook Structure

### Cell 1: Install Dependencies
Installs PyTorch 1.13.1+cu118 and all required packages for CUDA 11.8.

### Cell 2: Import Modules
Imports all necessary libraries and adds `./src/` to Python path.

### Cell 3: Configuration
Sets inference parameters:
- `checkpoint_path`: Path to model checkpoint
- `device`: CUDA or CPU
- `use_fp16`: Mixed precision inference
- `batch_size`: Batch size for image processing
- `video_frames`: Frames to extract per video
- `face_detector`: Face detection backend (mtcnn/retinaface/mediapipe)

### Cell 4: Validate Paths
Checks that checkpoint and data directory exist before proceeding.

### Cell 5: Initialize Inference Engine
Loads model checkpoint and creates inference engine with preprocessing pipeline.

### Cell 6: Run Inference
Processes all test files and generates predictions:
- Images: Direct inference on face crops
- Videos: Extract frames → inference → aggregate predictions

### Cell 7: Validate Submission
Verifies that `submission.csv` meets competition requirements:
- Correct columns: [filename, label]
- No null values
- Valid labels: {0, 1}
- No duplicate filenames

### Cell 8: Submit to AI Factory
Calls `aifactory.score.submit()` for automated scoring (AI Factory platform only).

## Configuration Options

### Inference Settings

Modify `CONFIG` in Cell 3 to adjust inference parameters:

```python
CONFIG = {
    "checkpoint_path": "checkpoints/best.pth",  # Model checkpoint
    "device": "cuda",                           # cuda or cpu
    "use_fp16": True,                          # Use FP16 for speed
    "batch_size": 64,                          # Image batch size
    "video_frames": 16,                        # Frames per video
    "face_detector": "mtcnn",                  # Face detector
}
```

### Performance Tuning

**For faster inference (GPU with ≥12GB VRAM)**:
```python
"use_fp16": True,
"batch_size": 128,
"video_frames": 32,
```

**For lower memory usage (GPU with <8GB VRAM)**:
```python
"use_fp16": True,
"batch_size": 32,
"video_frames": 8,
```

**For CPU inference**:
```python
"device": "cpu",
"use_fp16": False,
"batch_size": 16,
"video_frames": 8,
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce `batch_size` and/or `video_frames`:
```python
"batch_size": 32,
"video_frames": 8,
```

#### 2. Checkpoint Not Found
**Error**: `FileNotFoundError: Model checkpoint not found`

**Solution**: Ensure trained model is saved at the correct path:
```bash
# Check if checkpoint exists
ls -lh checkpoints/best.pth
```

If missing, train the model first:
```bash
python scripts/train.py --config configs/training_config.yaml --experiment my_model
```

#### 3. No Test Data
**Error**: `No test files found in data directory`

**Solution**: Verify test data is in `./data/`:
```bash
ls -lh ./data/
```

#### 4. Face Detection Fails
**Warning**: `No face detected in [file]`

**Solution**: The system uses fallback strategies:
- Try different face detectors: `mtcnn`, `retinaface`, `mediapipe`
- Uses full frame if no face detected (images)
- Skips problematic frames (videos)

#### 5. Import Errors
**Error**: `ModuleNotFoundError: No module named 'inference'`

**Solution**: Ensure `./src/` directory contains all source code:
```bash
# Check src structure
ls -R ./src/
```

Expected structure:
```
src/
├── inference/
│   ├── __init__.py
│   ├── inference_engine.py
│   └── model_loader.py
├── models/
├── data/
└── ...
```

## Performance Expectations

### Inference Speed (NVIDIA V100 16GB)
- Images: ~50-100 images/second
- Videos: ~5-10 videos/second
- Total time for 1000 files: ~5-10 minutes

### Inference Speed (NVIDIA T4 16GB)
- Images: ~30-50 images/second
- Videos: ~3-5 videos/second
- Total time for 1000 files: ~10-15 minutes

### Inference Speed (CPU)
- Images: ~1-2 images/second
- Videos: ~0.2-0.5 videos/second
- Total time for 1000 files: ~1-2 hours

## Validation Checks

The notebook automatically validates:

1. **Column names**: Must be exactly `["filename", "label"]`
2. **No null values**: All cells must have valid values
3. **Label values**: Must be 0 or 1 only
4. **Filename format**: Must have file extensions
5. **No duplicates**: Each filename must appear once
6. **Row count**: Must match number of test files

## Competition Metrics

**Evaluation Metric**: Macro F1-score

$$
\text{Macro F1} = \frac{F1_{\text{Real}} + F1_{\text{Fake}}}{2}
$$

Where:
- $F1_{\text{Real}}$ = F1-score for Real class (label=0)
- $F1_{\text{Fake}}$ = F1-score for Fake class (label=1)

**Target Performance**: Macro F1 ≥ 0.80

## Contact

For questions or issues:
- Competition Platform: AI Factory
- Competition: 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회

## License

This notebook is part of the competition submission and follows the competition rules and regulations.
