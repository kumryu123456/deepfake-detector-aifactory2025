# Model Training Guide

딥페이크 탐지 모델 훈련 가이드입니다.

## Overview

이 가이드는 실제 데이터셋으로 모델을 훈련하여 높은 성능(Macro F1 >80%)을 달성하는 방법을 설명합니다.

---

## Training Options

### Option 1: Baseline (빠른 시작) ⚡
**데이터**: FaceForensics++ only
**목표**: Macro F1 >80%
**시간**: ~2-4 hours (훈련만)
**권장**: 빠른 검증 및 첫 제출

### Option 2: Hybrid (높은 성능) 🎯
**데이터**: FaceForensics++ + DFDC + Celeb-DF
**목표**: Macro F1 >85%
**시간**: ~1-2 days (훈련만)
**권장**: 경쟁력 있는 최종 제출

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (권장: 12GB+)
- **RAM**: 16GB+ (32GB 권장)
- **Storage**: 100GB+ 여유 공간
- **CUDA**: 11.8

### Software Requirements
- Python 3.8+
- PyTorch 2.7.1+cu118
- All dependencies installed (requirements.txt)

**확인**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## Option 1: Baseline Training (FaceForensics++)

### Step 1: Download FaceForensics++ Dataset

#### Manual Download (Recommended)
1. GitHub 저장소: https://github.com/ondyari/FaceForensics
2. 다운로드 스크립트 실행:
```bash
# FaceForensics++ download script (from their repo)
python download-FaceForensics.py \
    -d FaceForensics++ \
    -c c23 \
    -t videos \
    --num_videos 1000
```

3. 데이터 구조:
```
data/faceforensics/
├── raw/
│   ├── real/          # Original videos
│   └── fake/          # Manipulated videos
│       ├── Deepfakes/
│       ├── Face2Face/
│       ├── FaceSwap/
│       └── NeuralTextures/
└── processed/         # Will be created by preprocessing
    ├── train/
    │   ├── real/
    │   └── fake/
    └── val/
        ├── real/
        └── fake/
```

**Note**: FaceForensics++ requires agreement to terms of use.

#### Alternative: Use Pre-downloaded Data
If you already have the dataset, organize it according to the structure above.

---

### Step 2: Preprocess Data

Face detection and cropping for all videos/images:

```bash
python scripts/preprocess_data.py \
    --input data/faceforensics/raw \
    --output data/faceforensics/processed \
    --detector mtcnn \
    --face-margin 0.2 \
    --image-size 224 \
    --num-workers 8 \
    --split-ratio 0.8
```

**Parameters**:
- `--detector`: Face detection method (mtcnn, retinaface, mediapipe)
- `--face-margin`: Margin around detected face (0.2 = 20%)
- `--image-size`: Output image size (224x224)
- `--num-workers`: Parallel processing workers
- `--split-ratio`: Train/val split (0.8 = 80% train, 20% val)

**Expected Output**:
```
Processing videos...
  Processed: 1000/1000 videos
  Faces detected: 945/1000 (94.5%)
  Train samples: 756
  Val samples: 189

Preprocessing complete!
  Output directory: data/faceforensics/processed/
  Total train images: 756 (real: 378, fake: 378)
  Total val images: 189 (real: 95, fake: 94)
```

**Time**: ~1-2 hours (depending on dataset size and hardware)

---

### Step 3: Verify Preprocessed Data

```bash
# Check data structure
ls -lh data/faceforensics/processed/train/
ls -lh data/faceforensics/processed/val/

# Count samples
find data/faceforensics/processed/train/real -type f | wc -l
find data/faceforensics/processed/train/fake -type f | wc -l
```

Expected:
- Balanced classes (similar number of real/fake)
- Images in JPEG format
- 224x224 resolution

---

### Step 4: Train Baseline Model

```bash
python scripts/train.py \
    --config configs/baseline_config.yaml \
    --experiment baseline_run1 \
    --gpu 0
```

**Training will**:
1. Load baseline_config.yaml
2. Create data loaders (80/20 split)
3. Initialize model (EfficientNet-B4 + dual-branch)
4. Train for 100 epochs (with early stopping)
5. Save checkpoints to `checkpoints/baseline_run1/`
6. Log to `logs/baseline_run1/`

**Expected Output**:
```
================================================================================
TRAINING CONFIGURATION
================================================================================
Experiment: baseline_run1
Model: DeepfakeDetector (43.6M parameters)
Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
Scheduler: CosineAnnealingLR (T_max=100, eta_min=1e-6)
Loss: CombinedLoss (CE:0.5, Focal:0.3, F1:0.2)
Batch size: 32
Mixed precision: Enabled
================================================================================

Epoch 1/100:
  Train Loss: 0.6234, Train Acc: 65.2%, Train F1: 0.6104
  Val Loss: 0.5123, Val Acc: 72.4%, Val F1: 0.7089
  Best F1! Saved checkpoint to checkpoints/baseline_run1/best.pth

Epoch 2/100:
  Train Loss: 0.4567, Train Acc: 78.3%, Train F1: 0.7756
  Val Loss: 0.3891, Val Acc: 82.1%, Val F1: 0.8145
  Best F1! Saved checkpoint to checkpoints/baseline_run1/best.pth

...

Epoch 45/100:
  Train Loss: 0.1234, Train Acc: 95.6%, Train F1: 0.9542
  Val Loss: 0.2145, Val Acc: 89.3%, Val F1: 0.8876
  No improvement for 15 epochs. Early stopping.

Training complete!
  Best validation F1: 0.8876
  Best checkpoint: checkpoints/baseline_run1/best.pth
  Training time: 2h 34m
```

**Time**: ~2-4 hours (depending on GPU)

**Monitor Training**:
```bash
# TensorBoard (in another terminal)
tensorboard --logdir logs/baseline_run1/

# Open browser: http://localhost:6006
```

---

### Step 5: Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/baseline_run1/best.pth \
    --data data/faceforensics/processed/val \
    --output results/baseline_evaluation.json
```

**Expected Output**:
```
================================================================================
EVALUATION RESULTS
================================================================================
Checkpoint: checkpoints/baseline_run1/best.pth
Dataset: data/faceforensics/processed/val/
Total samples: 189

Confusion Matrix:
                 Predicted
                Real    Fake
Actual  Real    90      5
        Fake    7       87

Metrics:
  Accuracy: 93.7%

  Real Class (0):
    Precision: 92.8%
    Recall: 94.7%
    F1-score: 93.7%

  Fake Class (1):
    Precision: 94.6%
    Recall: 92.6%
    F1-score: 93.6%

  Macro F1-score: 93.6% ✅

Per-sample predictions saved to: results/baseline_evaluation.json
================================================================================
```

**Target**: Macro F1 >80% ✅

---

### Step 6: Test Inference Speed

```bash
python scripts/inference.py \
    --checkpoint checkpoints/baseline_run1/best.pth \
    --data data/faceforensics/processed/val \
    --output submission_test.csv
```

**Verify**:
- Inference time < 3 hours for full test set
- submission_test.csv format correct

---

### Step 7: Update task.ipynb

Edit `task.ipynb` Cell 3:

```python
CONFIG = {
    "checkpoint_path": "checkpoints/baseline_run1/best.pth",  # ← Updated
    "data_dir": "./data",
    "output_path": "submission.csv",
    # ... rest of config
}
```

---

### Step 8: Test Notebook Locally

```bash
jupyter notebook task.ipynb
```

Run all cells and verify:
- ✅ Dependencies install
- ✅ Checkpoint loads
- ✅ Inference runs
- ✅ submission.csv generated
- ✅ Format validated

---

### Step 9: Submit to Competition

1. Upload to AI Factory:
   - `task.ipynb`
   - `checkpoints/baseline_run1/best.pth`
   - All files in `src/`

2. Run automated scoring

3. Check results

**Expected**: Macro F1 >80% on competition test set

---

## Option 2: Hybrid Training (Multi-Dataset)

### Additional Datasets

#### DFDC (Facebook Deepfake Detection Challenge)
- **Size**: ~124,000 videos
- **Link**: https://ai.facebook.com/datasets/dfdc/
- **Download**: Requires Kaggle account

```bash
# Via Kaggle API
kaggle competitions download -c deepfake-detection-challenge

# Extract to data/dfdc/raw/
```

#### Celeb-DF v2
- **Size**: 590 real + 5,639 fake videos
- **Link**: https://github.com/yuezunli/celeb-deepfakeforensics
- **Download**: Follow repo instructions

```bash
# Extract to data/celebdf/raw/
```

---

### Preprocess All Datasets

```bash
# FaceForensics++
python scripts/preprocess_data.py \
    --input data/faceforensics/raw \
    --output data/faceforensics/processed \
    --detector mtcnn

# DFDC
python scripts/preprocess_data.py \
    --input data/dfdc/raw \
    --output data/dfdc/processed \
    --detector mtcnn

# Celeb-DF
python scripts/preprocess_data.py \
    --input data/celebdf/raw \
    --output data/celebdf/processed \
    --detector mtcnn
```

---

### Update Hybrid Config

Edit `configs/hybrid_config.yaml` (or create from baseline_config.yaml):

```yaml
dataset:
  train_data:
    - type: "faceforensics"
      path: "data/faceforensics/processed/train"
    - type: "dfdc"
      path: "data/dfdc/processed/train"
    - type: "celebdf"
      path: "data/celebdf/processed/train"

  val_data:
    - type: "celebdf"  # Cross-dataset validation
      path: "data/celebdf/processed/val"

  sampling_strategy: "balanced"  # Equal sampling from each dataset
```

---

### Train Hybrid Model

```bash
python scripts/train.py \
    --config configs/hybrid_config.yaml \
    --experiment hybrid_run1 \
    --gpu 0
```

**Expected**:
- Higher generalization (cross-dataset)
- Macro F1 >85%
- Longer training time (~1-2 days)

---

## Troubleshooting

### CUDA Out of Memory

**Solution 1**: Reduce batch size
```yaml
# configs/baseline_config.yaml
training:
  batch_size: 16  # From 32
```

**Solution 2**: Enable gradient checkpointing
```yaml
memory:
  gradient_checkpointing: true
```

**Solution 3**: Reduce image size
```yaml
dataset:
  image_size: 192  # From 224
```

---

### Low Validation F1 (<70%)

**Possible causes**:
1. Class imbalance
2. Insufficient augmentation
3. Overfitting

**Solutions**:

**1. Check class balance**:
```bash
find data/*/processed/train/real -type f | wc -l
find data/*/processed/train/fake -type f | wc -l
```

Should be similar. If not, use weighted sampling:
```yaml
dataset:
  class_balance: true
  sampling_strategy: "balanced"
```

**2. Increase augmentation**:
```yaml
augmentation:
  enabled: true
  horizontal_flip: 0.5
  rotation: 15
  jpeg_compression:
    quality_lower: 60
    quality_upper: 100
    p: 0.5
```

**3. Add regularization**:
```yaml
optimizer:
  weight_decay: 0.02  # From 0.01

training:
  dropout: 0.2  # Add dropout
```

---

### Slow Training

**Solution 1**: Increase batch size (if memory allows)
```yaml
training:
  batch_size: 64  # From 32
```

**Solution 2**: Reduce workers (if CPU bottleneck)
```yaml
training:
  num_workers: 4  # From 8
```

**Solution 3**: Use distributed training (multi-GPU)
```bash
python scripts/train.py \
    --config configs/baseline_config.yaml \
    --experiment baseline_run1 \
    --gpu 0,1,2,3  # Use 4 GPUs
```

---

### Face Detection Failures

If many faces not detected (>10% failure rate):

**Solution**: Try different detector
```bash
python scripts/preprocess_data.py \
    --detector retinaface  # Instead of mtcnn
```

Or use ensemble:
```bash
python scripts/preprocess_data.py \
    --detector mtcnn \
    --fallback-detector mediapipe  # Use mediapipe if mtcnn fails
```

---

## Training Checklist

Before starting training:

- [ ] GPU available and CUDA working
- [ ] Dataset downloaded and extracted
- [ ] Data preprocessed (faces detected and cropped)
- [ ] Train/val split balanced
- [ ] Config file reviewed and customized
- [ ] Checkpoint directory exists
- [ ] Log directory exists
- [ ] Sufficient disk space (>50GB for checkpoints/logs)

During training:

- [ ] Monitor TensorBoard for metrics
- [ ] Check for overfitting (train F1 >> val F1)
- [ ] Watch GPU memory usage
- [ ] Verify checkpoints being saved

After training:

- [ ] Evaluate on validation set (F1 >80%)
- [ ] Test inference speed (<3 hours for test set)
- [ ] Update task.ipynb with best checkpoint
- [ ] Test notebook locally
- [ ] Submit to competition

---

## Performance Targets

### Baseline (FaceForensics++ only)
- **Training Set**: Accuracy >95%, F1 >0.94
- **Validation Set**: Accuracy >85%, F1 >0.82
- **Competition Test Set**: Macro F1 >80% ✅

### Hybrid (Multi-dataset)
- **Training Set**: Accuracy >92%, F1 >0.90
- **Validation Set** (Cross-dataset): Accuracy >82%, F1 >0.80
- **Competition Test Set**: Macro F1 >85% ✅

---

## Next Steps After Training

1. **Evaluate thoroughly**:
   - Validation set
   - Cross-dataset (Celeb-DF if trained on FF++)
   - Competition sample data

2. **Analyze errors**:
   - Which samples are misclassified?
   - Are there patterns? (lighting, angle, compression)

3. **Fine-tune if needed**:
   - Adjust loss weights (more emphasis on F1)
   - Train longer with lower learning rate
   - Add more augmentation

4. **Prepare final submission**:
   - Best checkpoint
   - Clean task.ipynb
   - Test on AI Factory platform

5. **Consider ensemble** (advanced):
   - Train multiple models
   - Average predictions
   - Often improves performance 2-3%

---

## Resources

- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **DFDC**: https://ai.facebook.com/datasets/dfdc/
- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics
- **Competition**: https://aifactory.space/task/9197
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **Timm Models**: https://github.com/huggingface/pytorch-image-models

---

## Questions?

Check:
1. **QUICKSTART.md** - Fast pipeline testing
2. **SETUP_GUIDE.md** - Environment setup
3. **README_NOTEBOOK.md** - task.ipynb usage
4. **PIPELINE_TEST_RESULTS.md** - Test results

Or refer to project README.md for contact information.

---

**Good luck with training!** 🚀
