#!/bin/bash
# Auto-accept FaceForensics TOS and download

set -euo pipefail

usage() {
    echo "Usage: $0 DATASET NUM_VIDEOS"
    exit 1
}

DATASET=${1:-}
NUM_VIDEOS=${2:-}

if [ -z "$DATASET" ] || [ -z "$NUM_VIDEOS" ]; then
    echo "Error: Missing required arguments."
    usage
fi

if ! [[ "$NUM_VIDEOS" =~ ^[0-9]+$ ]] || [ "$NUM_VIDEOS" -le 0 ]; then
    echo "Error: NUM_VIDEOS must be a positive integer."
    exit 1
fi

echo "Downloading $DATASET (target: $NUM_VIDEOS videos)..."

# Send enter key to accept TOS
if ! yes '' | python faceforensics_download_v4.py output_folder \
        -d "$DATASET" \
        -c raw \
        -t videos \
        -n "$NUM_VIDEOS" \
        --server EU; then
    echo "❌ Download failed for dataset '$DATASET'." >&2
    exit 1
fi
