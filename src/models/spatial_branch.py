"""Spatial branch for deepfake detection.

This module implements the spatial feature extraction branch using
EfficientNet backbone with Vision Transformer encoder.
"""

import timm
import torch
import torch.nn as nn
from typing import Optional


class SpatialBranch(nn.Module):
    """Spatial feature extraction branch.

    Uses EfficientNet backbone followed by Vision Transformer encoder
    to capture both local and global spatial features.

    Architecture:
        Input (B, 3, 224, 224)
        → EfficientNet-B4 → (B, 1792) features
        → Reshape to patches → (B, 49, 1792) for 7x7 patches
        → Add positional encoding
        → Vision Transformer (4 layers, 8 heads)
        → Global pooling → (B, 512) output features
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_dim: int = 512,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        output_dim: int = 512,
    ):
        """Initialize spatial branch.

        Args:
            backbone: Name of backbone architecture (from timm library)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            transformer_layers: Number of transformer encoder layers
            transformer_heads: Number of attention heads
            transformer_dim: Hidden dimension for transformer
            mlp_dim: MLP dimension in transformer
            dropout: Dropout rate
            output_dim: Output feature dimension
        """
        super().__init__()

        self.backbone_name = backbone
        self.output_dim = output_dim

        # Load EfficientNet backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="",  # Remove global pooling
        )

        # Get feature dimension from backbone
        # EfficientNet-B4 outputs (B, 1792, 7, 7) for 224x224 input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = self.backbone(dummy_input)
            self.feature_dim = backbone_output.shape[1]
            self.spatial_size = backbone_output.shape[2]  # e.g., 7 for 224x224 input
            self.num_patches = self.spatial_size ** 2  # e.g., 49

        # Freeze backbone if requested (for warmup phase)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection layer to match transformer dimension
        self.feature_projection = nn.Linear(self.feature_dim, transformer_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, transformer_dim) * 0.02
        )

        # Vision Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, output_dim),
            nn.ReLU(),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features.

        Args:
            x: RGB images, shape (batch_size, 3, H, W)

        Returns:
            Spatial features, shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Extract features with backbone
        # Output: (B, feature_dim, spatial_size, spatial_size)
        features = self.backbone(x)

        # Reshape to sequence of patches
        # (B, feature_dim, H, W) → (B, H*W, feature_dim)
        features = features.flatten(2).transpose(1, 2)  # (B, num_patches, feature_dim)

        # Project to transformer dimension
        features = self.feature_projection(features)  # (B, num_patches, transformer_dim)

        # Add positional encoding
        features = features + self.positional_encoding

        # Apply dropout
        features = self.dropout(features)

        # Pass through transformer
        features = self.transformer(features)  # (B, num_patches, transformer_dim)

        # Global average pooling across patches
        features = features.mean(dim=1)  # (B, transformer_dim)

        # Project to output dimension
        features = self.output_projection(features)  # (B, output_dim)

        return features

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

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
