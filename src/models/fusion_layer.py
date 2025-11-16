"""Fusion layer for combining spatial and frequency features.

This module implements attention-based fusion to adaptively combine
features from spatial and frequency branches.
"""

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """Feature fusion layer with self-attention.

    Combines spatial and frequency features using multi-head self-attention
    to learn optimal feature weighting and interactions.

    Architecture:
        Spatial (B, 512) + Frequency (B, 512) → Concatenate → (B, 1024)
        → Self-attention (4 heads)
        → Feed-forward network
        → Output (B, 1024) fused features
    """

    def __init__(
        self,
        input_dim: int = 1024,  # 512 spatial + 512 frequency
        attention_heads: int = 4,
        attention_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        """Initialize fusion layer.

        Args:
            input_dim: Combined feature dimension (spatial_dim + frequency_dim)
            attention_heads: Number of attention heads
            attention_dim: Dimension for attention computation
            hidden_dim: Hidden dimension for feed-forward network
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse spatial and frequency features.

        Args:
            spatial_features: Spatial features, shape (batch_size, spatial_dim)
            frequency_features: Frequency features, shape (batch_size, frequency_dim)

        Returns:
            Fused features, shape (batch_size, input_dim)
        """
        batch_size = spatial_features.shape[0]

        # Concatenate spatial and frequency features
        # (B, 512) + (B, 512) → (B, 1024)
        combined_features = torch.cat([spatial_features, frequency_features], dim=1)

        # Reshape for attention: (B, 1024) → (B, 1, 1024)
        # Treat combined features as a sequence of length 1
        features = combined_features.unsqueeze(1)  # (B, 1, 1024)

        # Self-attention with residual connection
        # Query, Key, Value are all the same (self-attention)
        attn_output, attn_weights = self.self_attention(
            features, features, features, need_weights=True
        )
        features = self.norm1(features + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(features)
        features = self.norm2(features + ff_output)

        # Remove sequence dimension: (B, 1, 1024) → (B, 1024)
        fused_features = features.squeeze(1)

        return fused_features

    def get_attention_weights(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
    ) -> torch.Tensor:
        """Get attention weights for visualization.

        Args:
            spatial_features: Spatial features
            frequency_features: Frequency features

        Returns:
            Attention weights, shape (batch_size, 1, 1)
        """
        batch_size = spatial_features.shape[0]

        # Concatenate features
        combined_features = torch.cat([spatial_features, frequency_features], dim=1)
        features = combined_features.unsqueeze(1)

        # Get attention weights
        with torch.no_grad():
            _, attn_weights = self.self_attention(
                features, features, features, need_weights=True
            )

        return attn_weights


class SimpleFusionLayer(nn.Module):
    """Simpler fusion layer using concatenation + FC layers.

    Alternative to attention-based fusion for faster inference.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        """Initialize simple fusion layer.

        Args:
            input_dim: Combined feature dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse spatial and frequency features.

        Args:
            spatial_features: Spatial features
            frequency_features: Frequency features

        Returns:
            Fused features
        """
        # Concatenate
        combined_features = torch.cat([spatial_features, frequency_features], dim=1)

        # Pass through fusion network with residual
        fused = self.fusion_network(combined_features)
        fused = self.norm(combined_features + fused)

        return fused


class GatedFusionLayer(nn.Module):
    """Gated fusion layer using learnable gates.

    Learns to weight spatial vs frequency features dynamically.
    """

    def __init__(
        self,
        spatial_dim: int = 512,
        frequency_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """Initialize gated fusion layer.

        Args:
            spatial_dim: Spatial feature dimension
            frequency_dim: Frequency feature dimension
            hidden_dim: Hidden dimension for gate computation
            dropout: Dropout rate
        """
        super().__init__()

        self.spatial_dim = spatial_dim
        self.frequency_dim = frequency_dim

        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(spatial_dim + frequency_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2 gates: one for spatial, one for frequency
            nn.Sigmoid(),  # Gate values in [0, 1]
        )

        # Feature projection (optional, to same dimension)
        self.spatial_proj = nn.Linear(spatial_dim, spatial_dim)
        self.frequency_proj = nn.Linear(frequency_dim, frequency_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse spatial and frequency features with learned gates.

        Args:
            spatial_features: Spatial features (B, spatial_dim)
            frequency_features: Frequency features (B, frequency_dim)

        Returns:
            Fused features (B, spatial_dim + frequency_dim)
        """
        # Compute gates
        combined = torch.cat([spatial_features, frequency_features], dim=1)
        gates = self.gate_network(combined)  # (B, 2)

        # Split gates
        spatial_gate = gates[:, 0:1]  # (B, 1)
        frequency_gate = gates[:, 1:2]  # (B, 1)

        # Apply gates
        weighted_spatial = spatial_features * spatial_gate
        weighted_frequency = frequency_features * frequency_gate

        # Project features
        spatial_proj = self.spatial_proj(weighted_spatial)
        frequency_proj = self.frequency_proj(weighted_frequency)

        # Concatenate weighted features
        fused = torch.cat([spatial_proj, frequency_proj], dim=1)
        fused = self.dropout(fused)

        return fused

    def get_gate_values(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get gate values for analysis.

        Args:
            spatial_features: Spatial features
            frequency_features: Frequency features

        Returns:
            Tuple of (spatial_gates, frequency_gates)
        """
        with torch.no_grad():
            combined = torch.cat([spatial_features, frequency_features], dim=1)
            gates = self.gate_network(combined)

        return gates[:, 0], gates[:, 1]
