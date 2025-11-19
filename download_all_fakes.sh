#!/bin/bash
# Sequential download of all Fake methods

set -euo pipefail

download_dataset() {
    local dataset=$1
    local videos=$2

    echo "=== Downloading ${dataset} (${videos} videos) ==="
    if ! yes '' | python faceforensics_download_v4.py output_folder \
            -d "$dataset" \
            -c raw \
            -t videos \
            -n "$videos" \
            --server EU2; then
        echo "❌ Download failed for dataset '$dataset'." >&2
        exit 1
    fi
}

download_dataset "Deepfakes" 350
download_dataset "Face2Face" 350
download_dataset "FaceSwap" 350

echo "=== All downloads complete! ==="
echo "Deepfakes: $(find output_folder/manipulated_sequences/Deepfakes -name "*.mp4" | wc -l)"
echo "Face2Face: $(find output_folder/manipulated_sequences/Face2Face -name "*.mp4" | wc -l)"
echo "FaceSwap: $(find output_folder/manipulated_sequences/FaceSwap -name "*.mp4" | wc -l)"
