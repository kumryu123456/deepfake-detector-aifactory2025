#!/bin/bash

echo "=== 비디오 파일 갯수 확인 ==="
echo ""
REAL_VIDEOS=$(find output_folder/original_sequences -name "*.mp4" 2>/dev/null | wc -l)
DEEPFAKES=$(find output_folder/manipulated_sequences/Deepfakes -name "*.mp4" 2>/dev/null | wc -l)
FACE2FACE=$(find output_folder/manipulated_sequences/Face2Face -name "*.mp4" 2>/dev/null | wc -l)
FACESWAP=$(find output_folder/manipulated_sequences/FaceSwap -name "*.mp4" 2>/dev/null | wc -l)
FAKE_TOTAL=$(find output_folder/manipulated_sequences -name "*.mp4" 2>/dev/null | wc -l)

echo "Real: $REAL_VIDEOS"
echo "Deepfakes: $DEEPFAKES"
echo "Face2Face: $FACE2FACE"
echo "FaceSwap: $FACESWAP"
echo "Fake 총합: $FAKE_TOTAL"

echo ""
echo "=== 전처리된 이미지 갯수 ==="
echo ""
TRAIN_REAL=$(find data/faceforensics/processed/train/real -name "*.jpg" 2>/dev/null | wc -l)
TRAIN_FAKE=$(find data/faceforensics/processed/train/fake -name "*.jpg" 2>/dev/null | wc -l)
VAL_REAL=$(find data/faceforensics/processed/val/real -name "*.jpg" 2>/dev/null | wc -l)
VAL_FAKE=$(find data/faceforensics/processed/val/fake -name "*.jpg" 2>/dev/null | wc -l)

echo "Train:"
echo "  Real: $TRAIN_REAL"
echo "  Fake: $TRAIN_FAKE"
echo "Validation:"
echo "  Real: $VAL_REAL"
echo "  Fake: $VAL_FAKE"

echo ""
echo "=== 비율 분석 ==="
echo ""
echo "비디오 비율: Real $REAL_VIDEOS : Fake $FAKE_TOTAL"
if [ $REAL_VIDEOS -gt 0 ]; then
  RATIO=$(awk "BEGIN {printf \"%.2f\", $FAKE_TOTAL / $REAL_VIDEOS}")
  echo "  → Real 1 : Fake $RATIO"
fi

echo ""
echo "Train 이미지 비율: Real $TRAIN_REAL : Fake $TRAIN_FAKE"
if [ $TRAIN_REAL -gt 0 ]; then
  TRAIN_RATIO=$(awk "BEGIN {printf \"%.2f\", $TRAIN_FAKE / $TRAIN_REAL}")
  echo "  → Real 1 : Fake $TRAIN_RATIO"
fi

echo ""
echo "Val 이미지 비율: Real $VAL_REAL : Fake $VAL_FAKE"
if [ $VAL_REAL -gt 0 ]; then
  VAL_RATIO=$(awk "BEGIN {printf \"%.2f\", $VAL_FAKE / $VAL_REAL}")
  echo "  → Real 1 : Fake $VAL_RATIO"
fi

echo ""
echo "=== 문제 분석 ==="
if [ $FAKE_TOTAL -gt $REAL_VIDEOS ]; then
  echo "⚠️  Fake 비디오가 Real보다 많습니다!"
  echo "   Real: $REAL_VIDEOS vs Fake: $FAKE_TOTAL"
fi

if [ $TRAIN_FAKE -gt $((TRAIN_REAL * 2)) ]; then
  echo "⚠️  Train Fake 이미지가 Real의 2배 이상입니다!"
  echo "   데이터 불균형이 심합니다."
fi
