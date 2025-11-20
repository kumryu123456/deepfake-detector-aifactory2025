#!/usr/bin/env python3
"""
Create Demo Checkpoint for Pipeline Testing

This script creates a demo model checkpoint with random initialization.
Use this to test the complete inference pipeline (task.ipynb) before
training on real data.

Usage:
    python scripts/create_demo_checkpoint.py --config configs/baseline_config.yaml --output checkpoints/demo.pth
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import argparse
import torch
import yaml
from datetime import datetime

# Import model
from models import create_model_from_config


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_demo_checkpoint(config_path: str, output_path: str):
    """Create a demo checkpoint with random initialization.

    Args:
        config_path: Path to configuration file
        output_path: Path to save checkpoint
    """
    print("=" * 80)
    print("CREATING DEMO CHECKPOINT")
    print("=" * 80)

    # Load config
    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)

    # Create model
    print(f"\nCreating model with config:")
    print(f"  Type: {config.get('model', {}).get('type', 'deepfake_detector')}")
    print(f"  Spatial backbone: {config.get('model', {}).get('spatial_branch', {}).get('backbone', 'efficientnet_b4')}")
    print(f"  Frequency branch: {config.get('model', {}).get('frequency_branch', {}).get('enabled', True)}")

    model = create_model_from_config(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)")

    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': 0,
        'best_metric': 0.0,
        'metrics': {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'val_accuracy': 0.0,
            'val_macro_f1': 0.0,
        },
        'created_at': datetime.now().isoformat(),
        'note': 'Demo checkpoint with random initialization for pipeline testing',
    }

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    print(f"\nSaving checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)

    # Verify checkpoint
    file_size = output_path.stat().st_size / 1024**2
    print(f"Checkpoint saved successfully!")
    print(f"  File size: {file_size:.2f} MB")

    # Test loading
    print(f"\nVerifying checkpoint can be loaded...")
    loaded = torch.load(output_path, map_location='cpu')
    assert 'model_state_dict' in loaded
    assert 'config' in loaded
    print(f"✅ Checkpoint verification passed!")

    print("\n" + "=" * 80)
    print("DEMO CHECKPOINT CREATED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Test inference pipeline:")
    print(f"   python scripts/inference.py --checkpoint {output_path} --data ./data --output submission.csv")
    print("\n2. Test submission notebook:")
    print(f"   jupyter notebook task.ipynb")
    print(f"   (Update checkpoint_path to '{output_path}')")
    print("\n3. When ready, train real model:")
    print(f"   python scripts/train.py --config {config_path} --experiment baseline")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Create demo checkpoint for pipeline testing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_config.yaml",
        help="Path to configuration file (default: configs/baseline_config.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/demo.pth",
        help="Output checkpoint path (default: checkpoints/demo.pth)"
    )

    args = parser.parse_args()

    # Validate config exists
    if not Path(args.config).exists():
        print(f"❌ Error: Config file not found: {args.config}")
        print("\nAvailable configs:")
        config_dir = Path("configs")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                print(f"  - {config_file}")
        sys.exit(1)

    try:
        create_demo_checkpoint(args.config, args.output)
    except Exception as e:
        print(f"\n❌ Error creating checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
