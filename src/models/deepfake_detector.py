"""Main deepfake detection model.

This module implements the complete dual-branch hybrid architecture
combining spatial and frequency domain analysis for deepfake detection.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .spatial_branch import SpatialBranch
from .frequency_branch import FrequencyBranch
from .fusion_layer import FusionLayer


class DeepfakeDetector(nn.Module):
    """Dual-branch hybrid model for deepfake detection.

    Combines spatial features (EfficientNet + Transformer) with
    frequency domain features (FFT/DCT analysis) for robust detection.

    Architecture:
        Input (B, 3, 224, 224)
        ├─ Spatial Branch → (B, 512)
        ├─ Frequency Branch → (B, 512)
        └─ Fusion Layer → (B, 1024)
           └─ Classifier → (B, 2) logits [Real, Fake]
    """

    def __init__(
        self,
        num_classes: int = 2,
        spatial_config: Optional[Dict] = None,
        frequency_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
    ):
        """Initialize deepfake detector.

        Args:
            num_classes: Number of output classes (default: 2 for binary)
            spatial_config: Configuration for spatial branch
            frequency_config: Configuration for frequency branch
            fusion_config: Configuration for fusion layer
            classifier_config: Configuration for classifier head
        """
        super().__init__()

        self.num_classes = num_classes

        # Default configurations
        spatial_config = spatial_config or {}
        frequency_config = frequency_config or {}
        fusion_config = fusion_config or {}
        classifier_config = classifier_config or {}

        # Spatial branch
        self.spatial_branch = SpatialBranch(
            backbone=spatial_config.get("backbone", "efficientnet_b4"),
            pretrained=spatial_config.get("pretrained", True),
            freeze_backbone=spatial_config.get("freeze_backbone", False),
            transformer_layers=spatial_config.get("transformer", {}).get("num_layers", 4),
            transformer_heads=spatial_config.get("transformer", {}).get("num_heads", 8),
            transformer_dim=spatial_config.get("transformer", {}).get("hidden_dim", 512),
            mlp_dim=spatial_config.get("transformer", {}).get("mlp_dim", 2048),
            dropout=spatial_config.get("transformer", {}).get("dropout", 0.1),
            output_dim=spatial_config.get("output_dim", 512),
        )

        # Frequency branch
        self.frequency_branch = FrequencyBranch(
            method=frequency_config.get("method", "fft"),
            conv_channels=frequency_config.get("conv_channels", [64, 128, 256]),
            kernel_sizes=frequency_config.get("kernel_sizes", [3, 3, 3]),
            output_dim=frequency_config.get("output_dim", 512),
        )

        # Fusion layer
        fusion_input_dim = (
            spatial_config.get("output_dim", 512) +
            frequency_config.get("output_dim", 512)
        )

        self.fusion_layer = FusionLayer(
            input_dim=fusion_input_dim,
            attention_heads=fusion_config.get("attention_heads", 4),
            attention_dim=fusion_config.get("attention_dim", 256),
            hidden_dim=fusion_config.get("hidden_dim", 512),
            dropout=fusion_config.get("dropout", 0.3),
        )

        # Classification head
        hidden_dims = classifier_config.get("hidden_dims", [256])
        dropout = classifier_config.get("dropout", 0.5)

        classifier_layers = []
        current_dim = fusion_input_dim

        for hidden_dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        # Final classification layer
        classifier_layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
               Normalized RGB images

        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
                   Raw logits before softmax
        """
        # Extract spatial features
        spatial_features = self.spatial_branch(x)  # (B, 512)

        # Extract frequency features
        frequency_features = self.frequency_branch(x)  # (B, 512)

        # Fuse features
        fused_features = self.fusion_layer(spatial_features, frequency_features)  # (B, 1024)

        # Classify
        logits = self.classifier(fused_features)  # (B, num_classes)

        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with probabilities.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            predictions: Predicted class labels (0 or 1), shape (batch_size,)
            probabilities: Class probabilities, shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate features for analysis.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            features: Dictionary containing:
                - 'spatial_features': Tensor of shape (batch_size, 512)
                - 'frequency_features': Tensor of shape (batch_size, 512)
                - 'fused_features': Tensor of shape (batch_size, 1024)
        """
        # Extract features from each branch
        spatial_features = self.spatial_branch(x)
        frequency_features = self.frequency_branch(x)
        fused_features = self.fusion_layer(spatial_features, frequency_features)

        return {
            "spatial_features": spatial_features,
            "frequency_features": frequency_features,
            "fused_features": fused_features,
        }

    def freeze_backbone(self):
        """Freeze spatial backbone for warmup phase."""
        self.spatial_branch.freeze_backbone()

    def unfreeze_backbone(self):
        """Unfreeze spatial backbone for fine-tuning."""
        self.spatial_branch.unfreeze_backbone()

    def get_num_parameters(self) -> int:
        """Get total number of parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_summary(self):
        """Print model architecture summary."""
        print("=" * 80)
        print("DeepfakeDetector Model Summary")
        print("=" * 80)

        print("\nSpatial Branch:")
        print(f"  Backbone: {self.spatial_branch.backbone_name}")
        print(f"  Feature dim: {self.spatial_branch.feature_dim}")
        print(f"  Output dim: {self.spatial_branch.output_dim}")
        print(f"  Parameters: {self.spatial_branch.get_num_parameters():,}")

        print("\nFrequency Branch:")
        print(f"  Method: {self.frequency_branch.method}")
        print(f"  Output dim: {self.frequency_branch.output_dim}")

        print("\nFusion Layer:")
        print(f"  Input dim: {self.fusion_layer.input_dim}")
        print(f"  Attention heads: {self.fusion_layer.attention_heads}")

        print("\nClassifier:")
        print(f"  Output classes: {self.num_classes}")

        print("\nTotal:")
        print(f"  Total parameters: {self.get_num_parameters():,}")
        print(f"  Trainable parameters: {self.get_num_trainable_parameters():,}")

        print("=" * 80)


def create_model_from_config(config: Dict) -> DeepfakeDetector:
    """Create model from configuration dictionary.

    Args:
        config: Configuration dictionary with model settings

    Returns:
        DeepfakeDetector instance

    Example:
        >>> config = {
        ...     "num_classes": 2,
        ...     "spatial_branch": {"backbone": "efficientnet_b4"},
        ...     "frequency_branch": {"method": "fft"},
        ... }
        >>> model = create_model_from_config(config)
    """
    return DeepfakeDetector(
        num_classes=config.get("num_classes", 2),
        spatial_config=config.get("spatial_branch", {}),
        frequency_config=config.get("frequency_branch", {}),
        fusion_config=config.get("fusion", {}),
        classifier_config=config.get("classifier", {}),
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters by component.

    Args:
        model: DeepfakeDetector model

    Returns:
        Dictionary with parameter counts per component
    """
    if not isinstance(model, DeepfakeDetector):
        raise TypeError("Model must be DeepfakeDetector instance")

    return {
        "spatial_branch": sum(p.numel() for p in model.spatial_branch.parameters()),
        "frequency_branch": sum(p.numel() for p in model.frequency_branch.parameters()),
        "fusion_layer": sum(p.numel() for p in model.fusion_layer.parameters()),
        "classifier": sum(p.numel() for p in model.classifier.parameters()),
        "total": model.get_num_parameters(),
        "trainable": model.get_num_trainable_parameters(),
    }
