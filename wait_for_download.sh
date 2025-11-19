#!/bin/bash
# Monitor Real video download progress

TARGET=350
LOG_FILE="download_real_350.log"

echo "Real 비디오 다운로드 모니터링 시작..."

while true; do
  # Count downloaded videos
  COUNT=$(find output_folder/original_sequences -name "*.mp4" 2>/dev/null | wc -l)
  
  # Get progress from log
  PROGRESS=$(tail -1 "$LOG_FILE" 2>/dev/null | grep -oP 'Progress: \K[0-9]+' || echo "0")
  
  echo "[$(date +%H:%M:%S)] 진행률: ${PROGRESS}%, 완료: ${COUNT}/${TARGET}"
  
  # Check if complete
  if [ "$COUNT" -ge "$TARGET" ]; then
    echo "다운로드 완료! Total: $COUNT videos"
    break
  fi
  
  # Wait 60 seconds before next check
  sleep 60
done
