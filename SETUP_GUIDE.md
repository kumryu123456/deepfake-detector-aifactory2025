# 환경 설정 가이드

딥페이크 탐지 모델 프로젝트의 환경 설정 방법입니다.

## 시스템 요구사항

- **Python**: 3.8 이상 (권장: 3.9)
- **CUDA**: 11.8 (GPU 사용 시)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (권장)
- **메모리**: 16GB+ RAM
- **디스크**: 100GB+ 여유 공간 (데이터셋 저장용)

## 1. Python 가상환경 생성

### Option A: Conda (권장)

```bash
# Conda 환경 생성
conda create -n deepfake python=3.9 -y
conda activate deepfake

# Conda base 패키지
conda install -y numpy scipy
```

### Option B: venv

```bash
# venv 환경 생성
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

## 2. PyTorch 설치

### GPU 버전 (CUDA 11.8)

```bash
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### CPU 버전 (테스트용)

```bash
pip install torch==1.13.1 torchvision==0.14.1
```

### 설치 확인

```python
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

예상 출력:
```
PyTorch 1.13.1+cu118
CUDA available: True
```

## 3. 프로젝트 의존성 설치

```bash
# requirements.txt 기반 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install timm==0.9.2
pip install opencv-python-headless==4.8.1.78
pip install albumentations==1.3.1
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install pyyaml==6.0.1
pip install tqdm==4.66.1
pip install facenet-pytorch==2.5.3
pip install mediapipe==0.10.3
pip install pytest==7.4.3
```

## 4. 설치 검증

### 전체 의존성 확인

```bash
python -c "
import torch
import torchvision
import timm
import cv2
import albumentations
import pandas
import sklearn
import yaml
import tqdm
import facenet_pytorch
import mediapipe

print('✅ All dependencies installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

### 모델 임포트 테스트

```bash
cd /mnt/c/Users/kumry/OneDrive/Desktop/4-2학기/고급인공신경망
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python -c "
from models import create_model_from_config
print('✅ Model imports working!')
"
```

## 5. Demo Checkpoint 생성

환경 설정이 완료되면 demo checkpoint를 생성하여 전체 파이프라인을 테스트할 수 있습니다.

```bash
# Demo checkpoint 생성 (랜덤 초기화)
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

Verifying checkpoint can be loaded...
✅ Checkpoint verification passed!

================================================================================
DEMO CHECKPOINT CREATED SUCCESSFULLY
================================================================================
```

## 6. 추론 파이프라인 테스트

Demo checkpoint로 전체 추론 파이프라인을 테스트합니다.

### 테스트 데이터 준비

```bash
# 테스트 데이터 디렉토리 생성
mkdir -p data

# 대회 샘플 데이터 다운로드 (수동)
# https://aifactory.space/task/9197/data
# 다운로드한 파일을 ./data/ 디렉토리에 압축 해제
```

### 추론 스크립트 실행

```bash
python scripts/inference.py \
    --checkpoint checkpoints/demo.pth \
    --data ./data \
    --output submission.csv \
    --use-fp16 \
    --batch-size 32
```

### submission.csv 검증

```bash
python scripts/test_submission.py \
    --submission submission.csv \
    --verbose
```

## 7. Jupyter Notebook 테스트

### Jupyter 설치

```bash
pip install jupyter notebook ipykernel
```

### task.ipynb 실행

```bash
# Jupyter Notebook 시작
jupyter notebook task.ipynb
```

또는

```bash
# Jupyter Lab 시작 (더 나은 UX)
pip install jupyterlab
jupyter lab task.ipynb
```

### Notebook에서 checkpoint 경로 수정

task.ipynb의 Cell 3 (Configuration)에서:

```python
CONFIG = {
    "checkpoint_path": "checkpoints/demo.pth",  # demo checkpoint 사용
    # ... 나머지 설정
}
```

## 8. 실제 모델 훈련 (선택사항)

### 데이터셋 다운로드

```bash
# FaceForensics++ (필수)
bash scripts/download_faceforensics.sh

# DFDC (권장)
bash scripts/download_dfdc.sh

# Celeb-DF (권장)
bash scripts/download_celebdf.sh
```

### 데이터 전처리

```bash
python scripts/preprocess_data.py \
    --input data/faceforensics/raw \
    --output data/faceforensics/processed \
    --detector mtcnn \
    --num-workers 8
```

### Baseline 모델 훈련

```bash
python scripts/train.py \
    --config configs/baseline_config.yaml \
    --experiment baseline_run1 \
    --gpu 0
```

### 모델 평가

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/baseline/best.pth \
    --data data/faceforensics/processed/val \
    --output results/baseline_evaluation.json
```

## 트러블슈팅

### CUDA Out of Memory

```bash
# batch_size 줄이기
python scripts/train.py --config configs/baseline_config.yaml --batch-size 16

# 또는 config 파일 수정
# configs/baseline_config.yaml:
#   training:
#     batch_size: 16  # 32에서 16으로
```

### Import Error: No module named 'src'

```bash
# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 또는 영구 설정 (.bashrc or .zshrc)
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/project/src"' >> ~/.bashrc
source ~/.bashrc
```

### Face Detection 실패

```bash
# 다른 detector 시도
python scripts/inference.py --face-detector retinaface  # mtcnn 대신
python scripts/inference.py --face-detector mediapipe  # 또는 mediapipe
```

### Slow Inference

```bash
# FP16 활성화
python scripts/inference.py --use-fp16 --batch-size 64

# 비디오 프레임 수 줄이기
python scripts/inference.py --video-frames 8  # 16에서 8로
```

## 다음 단계

환경 설정이 완료되면:

1. ✅ Demo checkpoint 생성
2. ✅ 추론 파이프라인 테스트
3. ✅ task.ipynb 실행 검증
4. 🔄 실제 데이터로 모델 훈련 (선택)
5. 🚀 대회 제출

## 참고 자료

- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
- [CUDA 설치 가이드](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [대회 플랫폼](https://aifactory.space/task/9197)
- [프로젝트 README](README.md)
- [Notebook 사용 가이드](README_NOTEBOOK.md)
