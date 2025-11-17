# Quick Start Guide

대회 제출 파이프라인을 빠르게 테스트하기 위한 가이드입니다.

## 개요

이 가이드는 실제 데이터 훈련 없이 **전체 제출 파이프라인을 테스트**하는 방법을 안내합니다.
Demo checkpoint를 사용하여 task.ipynb가 제대로 작동하는지 확인할 수 있습니다.

**소요 시간**: 약 20-30분 (환경 설정 포함)

## Step 1: 환경 설정 (10-15분)

### Option A: 자동 설치 스크립트 (권장)

```bash
# 프로젝트 디렉토리로 이동
cd /mnt/c/Users/kumry/OneDrive/Desktop/4-2학기/고급인공신경망

# 설치 스크립트 실행
bash scripts/setup_environment.sh
```

스크립트가 자동으로 다음을 수행합니다:
- Python 버전 확인 (>= 3.8)
- PyTorch 1.13.1+cu118 설치
- 프로젝트 의존성 설치
- 설치 검증

### Option B: 수동 설치

```bash
# 1. 가상환경 생성 (선택사항이지만 권장)
conda create -n deepfake python=3.9 -y
conda activate deepfake

# 2. PyTorch 설치 (CUDA 11.8)
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 3. 프로젝트 의존성 설치
pip install -r requirements.txt

# 4. 설치 확인
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

예상 출력:
```
PyTorch 1.13.1+cu118
CUDA: True
```

## Step 2: Demo Checkpoint 생성 (1-2분)

Demo checkpoint는 랜덤 초기화된 모델로, 파이프라인 테스트용입니다.

```bash
# Demo checkpoint 생성
python scripts/create_demo_checkpoint.py \
    --config configs/baseline_config.yaml \
    --output checkpoints/demo.pth
```

예상 출력:
```
================================================================================
CREATING DEMO CHECKPOINT
================================================================================

Loading config from: configs/baseline_config.yaml

Creating model with config:
  Type: deepfake_detector
  Spatial backbone: efficientnet_b4
  Frequency branch: True

Model statistics:
  Total parameters: 22,451,234
  Trainable parameters: 22,451,234
  Model size: 85.65 MB (FP32)

Saving checkpoint to: checkpoints/demo.pth
Checkpoint saved successfully!
  File size: 85.87 MB

✅ Checkpoint verification passed!
================================================================================
```

## Step 3: 테스트 데이터 준비 (2-3분)

### Option A: 대회 샘플 데이터 다운로드

1. https://aifactory.space/task/9197/data 접속
2. 샘플 데이터 다운로드 (7 fake images + 5 fake videos)
3. `./data/` 디렉토리에 압축 해제

```bash
# 디렉토리 생성
mkdir -p data

# 다운로드한 파일을 data/로 이동
# 예: mv ~/Downloads/sample_data/* ./data/
```

### Option B: 더미 데이터 생성 (테스트용)

실제 데이터가 없다면 더미 이미지로 테스트:

```bash
# 더미 이미지 생성 (Python으로)
python -c "
from PIL import Image
import numpy as np
from pathlib import Path

Path('data').mkdir(exist_ok=True)

# 더미 이미지 5개 생성
for i in range(5):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(f'data/test_image_{i}.jpg')

print('✅ Created 5 dummy test images')
"
```

## Step 4: 추론 파이프라인 테스트 (1-2분)

CLI 스크립트로 추론 테스트:

```bash
python scripts/inference.py \
    --checkpoint checkpoints/demo.pth \
    --data ./data \
    --output submission.csv \
    --use-fp16 \
    --batch-size 32
```

예상 출력:
```
================================================================================
INFERENCE CONFIGURATION
================================================================================
  checkpoint_path: checkpoints/demo.pth
  data_dir: ./data
  device: cuda
  use_fp16: True
  batch_size: 32
================================================================================

Loading checkpoint...
✅ Checkpoint loaded successfully

Initializing inference engine...
✅ Inference engine initialized

Scanning data directory...
Found 5 files (5 images, 0 videos)

Running inference...
Processing: 100%|████████████████████| 5/5 [00:02<00:00, 2.31 files/s]

================================================================================
INFERENCE COMPLETED SUCCESSFULLY!
================================================================================
Total time: 2.16 seconds
Average time per file: 0.432 seconds

Prediction statistics:
  Total predictions: 5
  Real (0): 2 (40.0%)
  Fake (1): 3 (60.0%)

Output saved to: submission.csv
================================================================================
```

## Step 5: Submission 검증 (< 1분)

생성된 submission.csv 검증:

```bash
python scripts/test_submission.py \
    --submission submission.csv \
    --verbose
```

예상 출력:
```
================================================================================
VALIDATING SUBMISSION
================================================================================

✅ All validation checks passed!

Submission summary:
  Total rows: 5
  Real (0): 2
  Fake (1): 3
================================================================================
```

## Step 6: Jupyter Notebook 테스트 (5-10분)

### 6.1 Jupyter 설치 (처음 한 번만)

```bash
pip install jupyter notebook
```

### 6.2 Notebook 실행

```bash
jupyter notebook task.ipynb
```

브라우저가 자동으로 열립니다.

### 6.3 Cell 3 수정

Checkpoint 경로를 demo checkpoint로 변경:

```python
CONFIG = {
    "checkpoint_path": "checkpoints/demo.pth",  # ← demo checkpoint 사용
    "data_dir": "./data",
    "output_path": "submission.csv",
    # ... 나머지 설정
}
```

### 6.4 모든 Cell 실행

- 메뉴: `Kernel` → `Restart & Run All`
- 또는 각 Cell을 순차적으로 실행 (`Shift + Enter`)

### 6.5 출력 확인

마지막 Cell에서 다음과 같은 출력을 확인:

```
================================================================================
INFERENCE COMPLETED SUCCESSFULLY!
================================================================================
Total time: 2.16 seconds

Prediction statistics:
  Total predictions: 5
  Real (0): 2
  Fake (1): 3

Output saved to: submission.csv
================================================================================

Validating submission format...
================================================================================

✅ All validation checks passed!

Submission summary:
  Total rows: 5
  Real (0): 2
  Fake (1): 3
================================================================================
```

## 완료! 🎉

전체 제출 파이프라인이 정상 작동합니다!

### 다음 단계

#### Option A: 대회에 바로 제출

Demo checkpoint로도 제출 가능합니다 (성능은 낮지만 파이프라인 검증용):

1. task.ipynb를 AI Factory 플랫폼에 업로드
2. checkpoints/demo.pth도 함께 업로드
3. 자동 채점 실행

#### Option B: 실제 모델 훈련 후 제출

더 나은 성능을 위해 실제 데이터로 훈련:

```bash
# 1. 데이터셋 다운로드
bash scripts/download_faceforensics.sh

# 2. 데이터 전처리
python scripts/preprocess_data.py \
    --input data/faceforensics/raw \
    --output data/faceforensics/processed

# 3. 모델 훈련
python scripts/train.py \
    --config configs/baseline_config.yaml \
    --experiment baseline_run1

# 4. 최고 성능 checkpoint 사용
# task.ipynb의 checkpoint_path를 변경:
# "checkpoints/baseline/best.pth"

# 5. task.ipynb 재실행 및 제출
```

## 트러블슈팅

### CUDA Out of Memory

```python
# task.ipynb Cell 3에서 batch_size 줄이기
CONFIG = {
    # ...
    "batch_size": 16,  # 32에서 16으로
    # ...
}
```

### Face Detection 실패

```python
# task.ipynb Cell 3에서 다른 detector 시도
CONFIG = {
    # ...
    "face_detector": "mediapipe",  # mtcnn 대신
    # ...
}
```

### Import Error

```bash
# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 또는 .bashrc에 추가 (영구 설정)
echo 'export PYTHONPATH="${PYTHONPATH}:'$(pwd)'/src"' >> ~/.bashrc
source ~/.bashrc
```

## 참고 자료

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - 상세한 환경 설정 가이드
- [README_NOTEBOOK.md](README_NOTEBOOK.md) - Notebook 사용 가이드
- [README.md](README.md) - 프로젝트 개요
- [대회 플랫폼](https://aifactory.space/task/9197)

## 요약

```bash
# 1. 환경 설정
bash scripts/setup_environment.sh

# 2. Demo checkpoint 생성
python scripts/create_demo_checkpoint.py --config configs/baseline_config.yaml --output checkpoints/demo.pth

# 3. 추론 테스트
python scripts/inference.py --checkpoint checkpoints/demo.pth --data ./data --output submission.csv

# 4. 검증
python scripts/test_submission.py --submission submission.csv

# 5. Notebook 테스트
jupyter notebook task.ipynb
```

**총 소요 시간**: 20-30분

파이프라인이 작동하면 실제 모델을 훈련하고 checkpoint만 교체하면 됩니다! 🚀
