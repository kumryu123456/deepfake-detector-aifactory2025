#!/bin/bash
# 전처리 완료 확인 및 mp4 삭제 스크립트

DEFAULT_MIN_IMAGES=1000
CONFIGURED_MIN_IMAGES="${MIN_IMAGES:-}"

if [ -n "$1" ]; then
    CONFIGURED_MIN_IMAGES="$1"
fi

if [ -z "$CONFIGURED_MIN_IMAGES" ]; then
    CONFIGURED_MIN_IMAGES=$DEFAULT_MIN_IMAGES
fi

if ! [[ "$CONFIGURED_MIN_IMAGES" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ Invalid MIN_IMAGES value '$CONFIGURED_MIN_IMAGES'. Please provide a positive integer."
    exit 1
fi

MIN_IMAGES=$CONFIGURED_MIN_IMAGES

echo "Checking preprocessing status..."
if [ ! -f preprocessing.log ]; then
    echo "❌ preprocessing.log not found. Please run preprocessing before cleanup."
    exit 1
fi

if ! tail -5 preprocessing.log | grep -q "PREPROCESSING COMPLETE"; then
    echo "❌ Preprocessing not complete yet. Wait and try again."
    exit 1
fi

echo "Checking processed images..."
TRAIN_REAL=$(find data/faceforensics/processed/train/real -name "*.jpg" 2>/dev/null | wc -l)
TRAIN_FAKE=$(find data/faceforensics/processed/train/fake -name "*.jpg" 2>/dev/null | wc -l)
VAL_REAL=$(find data/faceforensics/processed/val/real -name "*.jpg" 2>/dev/null | wc -l)
VAL_FAKE=$(find data/faceforensics/processed/val/fake -name "*.jpg" 2>/dev/null | wc -l)

echo "Processed images:"
echo "  Train Real: $TRAIN_REAL"
echo "  Train Fake: $TRAIN_FAKE"
echo "  Val Real: $VAL_REAL"
echo "  Val Fake: $VAL_FAKE"
TOTAL_IMAGES=$((TRAIN_REAL + TRAIN_FAKE + VAL_REAL + VAL_FAKE))
echo "  Total: $TOTAL_IMAGES"

if [ "$TOTAL_IMAGES" -lt "$MIN_IMAGES" ]; then
    echo "❌ Too few images ($TOTAL_IMAGES found, need at least $MIN_IMAGES). Preprocessing may have failed."
    exit 1
fi

echo "✅ Preprocessing looks good. Ready to delete mp4 files."
echo ""
read -p "Delete output_folder/ (290GB)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting output_folder/..."
    rm -rf output_folder/
    echo "✅ Deleted. Checking disk space..."
    if grep -qi microsoft /proc/version 2>/dev/null; then
        df -h /mnt/c
    else
        df -h .
    fi
else
    echo "Cancelled."
fi
