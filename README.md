# Deepfake Detection AI Competition

국가수사본부 딥페이크 범죄 대응 AI 탐지 모델 경진대회 제출 프로젝트

## 프로젝트 개요

얼굴 이미지 및 동영상에서 실제(Real)와 가짜(Fake)를 판별하는 이진 분류 모델을 개발합니다.

**주요 특징:**
- 이중 브랜치 하이브리드 아키텍처 (공간 특징 + 주파수 도메인 분석)
- EfficientNet-B4 백본 + Vision Transformer
- 다중 데이터셋 훈련 (FaceForensics++, DFDC, Celeb-DF)
- Macro F1-score 최적화
- 3시간 이내 추론 시간 제약 준수

## 환경 설정

### 필수 요구사항
- Python 3.9
- CUDA 11.8
- GPU: NVIDIA GPU with 8GB+ VRAM

### 설치 방법

```bash
# 1. 가상환경 생성
conda create -n deepfake python=3.9
conda activate deepfake

# 2. PyTorch 설치 (CUDA 11.8)
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. 의존성 설치
pip install -r requirements.txt
```

## 프로젝트 구조

```
deepfake-detection/
├── src/                    # 소스 코드
│   ├── models/            # 모델 아키텍처
│   ├── data/              # 데이터 처리
│   ├── training/          # 훈련 컴포넌트
│   ├── inference/         # 추론 파이프라인
│   └── utils/             # 유틸리티
├── configs/               # 설정 파일
├── scripts/               # 실행 스크립트
├── notebooks/             # Jupyter 노트북
├── tests/                 # 테스트
├── data/                  # 데이터셋 (gitignore)
├── checkpoints/           # 모델 체크포인트 (gitignore)
└── logs/                  # 로그 (gitignore)
```

## 데이터셋 준비

### 대회 샘플 데이터
- 7개 가짜 이미지
- 5개 가짜 비디오
- 다운로드: https://aifactory.space/task/9197/data

### 훈련 데이터셋
1. **FaceForensics++** (필수)
   - 1,000 실제 + 4,000 가짜 비디오
   - https://github.com/ondyari/FaceForensics

2. **DFDC** (권장)
   - 124,000 비디오
   - https://ai.facebook.com/datasets/dfdc/

3. **Celeb-DF v2** (권장)
   - 590 실제 + 5,639 가짜 비디오
   - https://github.com/yuezunli/celeb-deepfakeforensics

## 사용 방법

### 훈련

```bash
# 기본 모델 훈련
python scripts/train.py --config configs/training_config.yaml

# 하이브리드 모델 훈련 (다중 데이터셋)
python scripts/train.py --config configs/hybrid_training_config.yaml
```

### 평가

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data data/validation/
```

### 추론

```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth --data ./data/ --output submission.csv
```

### 제출

`notebooks/task.ipynb` 노트북을 실행하여 대회에 제출합니다.

## 성능 목표

- **Baseline**: Macro F1-score >85% (FaceForensics++ test set)
- **Target**: Macro F1-score >82% (competition test set)
- **Inference**: <2 hours for ~10,000 samples

## 개발 타임라인

- **Week 1-2**: Phase 1-3 (기본 모델 개발)
- **Week 3-4**: Phase 4 (제출 파이프라인)
- **Week 5**: Phase 5 (하이브리드 모델)
- **Week 6**: Phase 6 (최종 최적화)

## 참고 자료

- [구현 계획](specs/001-deepfake-detection-competition/plan.md)
- [연구 자료](specs/001-deepfake-detection-competition/research.md)
- [빠른 시작 가이드](specs/001-deepfake-detection-competition/quickstart.md)
- [작업 목록](specs/001-deepfake-detection-competition/tasks.md)

## 대회 정보

- **주최**: 국가수사본부 (National Forensic Service)
- **마감**: 2025년 11월 20일 17:00
- **평가 지표**: Macro F1-score
- **제약사항**: 단일 모델, 3시간 추론 시간 제한

## 라이선스

본 프로젝트는 대회 참가용으로 개발되었으며, 대회 규정에 따라 상위 입상 시 기술 이전 계약이 적용됩니다.
