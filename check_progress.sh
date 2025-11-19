#!/bin/bash
# Quick progress check script

echo "=== 현재 진행 상황 ===" 
echo ""
echo "[1] Real 비디오 다운로드:"
REAL_COUNT=$(find output_folder/original_sequences -name "*.mp4" 2>/dev/null | wc -l)
echo "  완료: $REAL_COUNT/350"

echo ""
echo "[2] Fake 비디오 (Deepfakes):"
FAKE_COUNT=$(find output_folder/manipulated_sequences/Deepfakes -name "*.mp4" 2>/dev/null | wc -l)
echo "  완료: $FAKE_COUNT/357 (350개 사용 예정)"

echo ""
echo "[3] 전처리 데이터:"
if [ -d "data/faceforensics/processed/train" ]; then
  TRAIN_REAL=$(find data/faceforensics/processed/train/real -name "*.jpg" 2>/dev/null | wc -l)
  TRAIN_FAKE=$(find data/faceforensics/processed/train/fake -name "*.jpg" 2>/dev/null | wc -l)
  echo "  Train Real: $TRAIN_REAL images"
  echo "  Train Fake: $TRAIN_FAKE images"
else
  echo "  아직 시작 안됨"
fi

echo ""
echo "[4] 학습 진행:"
if [ -f "training_balanced.log" ]; then
  LAST_EPOCH=$(grep -oP 'Epoch \K[0-9]+' training_balanced.log | tail -1 || echo "0")
  echo "  현재 Epoch: $LAST_EPOCH/100"
else
  echo "  아직 시작 안됨"
fi

echo ""
echo "=== 로그 파일 ===" 
echo "  다운로드: tail -f download_real_350.log"
echo "  전처리: tail -f preprocessing_balanced.log"
echo "  학습: tail -f training_balanced.log"
echo "  자동 파이프라인: tail -f auto_pipeline.log"
