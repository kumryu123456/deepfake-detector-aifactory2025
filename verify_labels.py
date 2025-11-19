#!/usr/bin/env python3
"""Verify label CSV files and actual images match"""

import csv
from pathlib import Path
from collections import Counter

def check_labels(csv_path, data_dir):
    """Check labels and verify files exist"""
    # Read CSV
    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['filename']] = int(row['label'])
    
    # Count labels
    label_counts = Counter(labels.values())
    
    # Check which files actually exist
    data_path = Path(data_dir)
    existing_files = set(f.name for f in data_path.glob("*.jpg"))
    
    # Count how many labeled files exist
    existing_with_labels = {filename: label for filename, label in labels.items() if filename in existing_files}
    existing_counts = Counter(existing_with_labels.values())
    
    print(f"CSV: {csv_path}")
    print(f"Total labels in CSV: {len(labels)}")
    print(f"  Real (0): {label_counts[0]}")
    print(f"  Fake (1): {label_counts[1]}")
    print(f"\nActual files in {data_dir}: {len(existing_files)}")
    print(f"Files with labels: {len(existing_with_labels)}")
    print(f"  Real (0): {existing_counts[0]}")
    print(f"  Fake (1): {existing_counts[1]}")
    
    if label_counts[0] == label_counts[1]:
        print(f"\n✅ Labels are BALANCED: 1:1 ratio")
    else:
        ratio = label_counts[1] / label_counts[0] if label_counts[0] > 0 else 0
        print(f"\n⚠️  Labels are IMBALANCED: 1:{ratio:.2f}")
    
    return label_counts, existing_counts

print("=" * 80)
print("TRAIN DATASET")
print("=" * 80)
check_labels("data/faceforensics/processed/train_labels.csv", "data/faceforensics/processed/train")

print("\n" + "=" * 80)
print("VALIDATION DATASET")
print("=" * 80)
check_labels("data/faceforensics/processed/val_labels.csv", "data/faceforensics/processed/val")
