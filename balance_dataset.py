#!/usr/bin/env python3
"""Balance the dataset to have equal Real:Fake ratio"""

import os
import random
from pathlib import Path

def balance_images(real_dir, fake_dir, seed=42):
    """Balance by randomly removing excess fake images"""
    random.seed(seed)
    
    # Get all image files
    real_images = list(Path(real_dir).glob("*.jpg"))
    fake_images = list(Path(fake_dir).glob("*.jpg"))
    
    real_count = len(real_images)
    fake_count = len(fake_images)
    
    print(f"Before balancing:")
    print(f"  Real: {real_count}")
    print(f"  Fake: {fake_count}")
    print(f"  Ratio: 1:{fake_count/real_count:.2f}")
    
    if fake_count <= real_count:
        print("Already balanced or Fake is less. No action needed.")
        return
    
    # Calculate how many to remove
    to_remove = fake_count - real_count
    
    # Randomly select images to remove
    images_to_remove = random.sample(fake_images, to_remove)
    
    # Remove selected images
    for img in images_to_remove:
        img.unlink()
    
    # Verify
    remaining_fake = len(list(Path(fake_dir).glob("*.jpg")))
    
    print(f"\nAfter balancing:")
    print(f"  Real: {real_count}")
    print(f"  Fake: {remaining_fake}")
    print(f"  Removed: {to_remove}")
    print(f"  Ratio: 1:{remaining_fake/real_count:.2f}")

if __name__ == "__main__":
    print("=" * 80)
    print("BALANCING TRAIN DATASET")
    print("=" * 80)
    balance_images(
        "data/faceforensics/processed/train/real",
        "data/faceforensics/processed/train/fake"
    )
    
    print("\n" + "=" * 80)
    print("BALANCING VALIDATION DATASET")
    print("=" * 80)
    balance_images(
        "data/faceforensics/processed/val/real",
        "data/faceforensics/processed/val/fake"
    )
    
    print("\n" + "=" * 80)
    print("DATASET BALANCING COMPLETE!")
    print("=" * 80)
