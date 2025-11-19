#!/bin/bash
# Automatic pipeline execution after download completes

set -e  # Exit on error

echo "=== 다운로드 완료 대기 중 ==="

# Wait for 350 Real videos
while true; do
  COUNT=$(find output_folder/original_sequences -name "*.mp4" 2>/dev/null | wc -l)
  if [ "$COUNT" -ge 350 ]; then
    echo "Real 비디오 350개 다운로드 완료!"
    break
  fi
  echo "현재: $COUNT/350..."
  sleep 120  # Check every 2 minutes
done

echo ""
echo "=== 1단계: Fake 비디오 선택 (Deepfakes 350개만) ==="
# Already have 357 Deepfakes, so we're good

echo ""
echo "=== 2단계: 데이터 전처리 시작 ==="
source venv/bin/activate

# Delete old processed data
if [ -d "data/faceforensics/processed" ]; then
  echo "기존 전처리 데이터 삭제..."
  rm -rf data/faceforensics/processed
fi

# Create directory structure
mkdir -p data/faceforensics/processed

# Run preprocessing with only Real and Deepfakes
echo "전처리 실행 중..."
python scripts/preprocess_faceforensics.py \
  --input output_folder \
  --output data/faceforensics/processed \
  --detector mtcnn \
  --max-frames 10 \
  --val-split 0.2 \
  --device cuda \
  2>&1 | tee preprocessing_balanced.log

echo ""
echo "=== 전처리 완료! ==="

# Check results
echo "전처리 결과:"
find data/faceforensics/processed -type d -name "real" -o -name "fake" | while read dir; do
  count=$(find "$dir" -name "*.jpg" 2>/dev/null | wc -l)
  echo "  $dir: $count images"
done

echo ""
echo "=== 3단계: 학습 시작 ==="
python scripts/train.py \
  --config configs/baseline_config.yaml \
  --train-data data/faceforensics/processed/train \
  --train-labels data/faceforensics/processed/train_labels.csv \
  --val-data data/faceforensics/processed/val \
  --val-labels data/faceforensics/processed/val_labels.csv \
  --device cuda \
  2>&1 | tee training_balanced.log

echo ""
echo "=== 모든 단계 완료! ==="
