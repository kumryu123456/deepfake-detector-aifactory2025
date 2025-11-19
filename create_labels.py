#!/usr/bin/env python3
"""Create label CSV files for train and validation datasets"""

import csv
from pathlib import Path

def create_label_csv(data_dir, output_csv):
    """Create CSV with filename,label pairs"""
    data_path = Path(data_dir)
    
    # Collect all images with labels
    labels = []
    
    # Real images (label = 0)
    real_dir = data_path / "real"
    if real_dir.exists():
        for img in sorted(real_dir.glob("*.jpg")):
            labels.append((img.name, 0))
    
    # Fake images (label = 1)
    fake_dir = data_path / "fake"
    if fake_dir.exists():
        for img in sorted(fake_dir.glob("*.jpg")):
            labels.append((img.name, 1))
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(labels)
    
    # Print stats
    real_count = sum(1 for _, label in labels if label == 0)
    fake_count = sum(1 for _, label in labels if label == 1)
    
    print(f"Created {output_csv}")
    print(f"  Total: {len(labels)}")
    print(f"  Real (0): {real_count}")
    print(f"  Fake (1): {fake_count}")
    print(f"  Ratio: 1:{fake_count/real_count:.2f}" if real_count > 0 else "  Ratio: N/A")

if __name__ == "__main__":
    print("=" * 80)
    print("CREATING LABEL CSV FILES")
    print("=" * 80)
    print()
    
    # Create train labels
    create_label_csv(
        "data/faceforensics/processed/train",
        "data/faceforensics/processed/train_labels.csv"
    )
    
    print()
    
    # Create val labels
    create_label_csv(
        "data/faceforensics/processed/val",
        "data/faceforensics/processed/val_labels.csv"
    )
    
    print()
    print("=" * 80)
    print("LABEL CSV FILES CREATED!")
    print("=" * 80)
