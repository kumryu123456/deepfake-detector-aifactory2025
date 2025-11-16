"""Frequency branch for deepfake detection.

This module implements frequency domain feature extraction using
FFT/DCT to capture high-frequency artifacts typical of deepfakes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FrequencyBranch(nn.Module):
    """Frequency domain feature extraction branch.

    Applies FFT/DCT and processes amplitude/phase spectra with CNN
    to detect high-frequency artifacts from GAN upsampling.

    Architecture:
        Input (B, 3, 224, 224)
        → FFT/DCT → Amplitude + Phase spectra
        → Stack (B, 2*3, 224, 224) = 6 channels
        → Conv layers (64 → 128 → 256)
        → Adaptive pooling → FC → (B, 512) output features
    """

    def __init__(
        self,
        method: str = "fft",
        conv_channels: list = [64, 128, 256],
        kernel_sizes: list = [3, 3, 3],
        output_dim: int = 512,
    ):
        """Initialize frequency branch.

        Args:
            method: Frequency transform method ("fft" or "dct")
            conv_channels: List of channel dimensions for conv layers
            kernel_sizes: List of kernel sizes for conv layers
            output_dim: Output feature dimension
        """
        super().__init__()

        self.method = method.lower()
        self.output_dim = output_dim

        if self.method not in ["fft", "dct"]:
            raise ValueError(f"Invalid method: {method}. Must be 'fft' or 'dct'")

        # Input channels: 6 (amplitude + phase for each of R, G, B channels)
        in_channels = 6

        # Build convolutional layers
        conv_layers = []
        current_channels = in_channels

        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Calculate flattened dimension after pooling
        # Last conv channel × 7 × 7
        flatten_dim = conv_channels[-1] * 7 * 7

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def compute_frequency_transform(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute frequency domain representation.

        Args:
            x: Input images, shape (batch_size, 3, H, W)

        Returns:
            amplitude: Amplitude spectrum, shape (batch_size, 3, H, W)
            phase: Phase spectrum, shape (batch_size, 3, H, W)
        """
        if self.method == "fft":
            return self._compute_fft(x)
        elif self.method == "dct":
            return self._compute_dct(x)

    def _compute_fft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D FFT.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            amplitude, phase
        """
        # Apply 2D FFT on spatial dimensions (H, W)
        # fft_result is complex tensor
        fft_result = torch.fft.fft2(x, dim=(-2, -1))

        # Shift zero frequency to center
        fft_result = torch.fft.fftshift(fft_result, dim=(-2, -1))

        # Compute amplitude and phase
        amplitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)

        # Log scale for amplitude (better visualization and features)
        # Add small epsilon to avoid log(0)
        amplitude = torch.log(amplitude + 1e-8)

        return amplitude, phase

    def _compute_dct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D DCT (approximation using FFT).

        Args:
            x: Input images (B, C, H, W)

        Returns:
            amplitude, phase (phase is zeros for DCT)
        """
        # DCT can be computed using FFT of a symmetrically extended signal
        # For simplicity, we use FFT and take real part as DCT approximation

        # Apply FFT
        fft_result = torch.fft.fft2(x, dim=(-2, -1))

        # DCT-like: use real part and magnitude
        dct_approx = torch.real(fft_result)

        # Apply log scaling
        amplitude = torch.log(torch.abs(dct_approx) + 1e-8)

        # Phase is not meaningful for DCT, use zeros
        phase = torch.zeros_like(amplitude)

        return amplitude, phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frequency domain features.

        Args:
            x: RGB images, shape (batch_size, 3, H, W)

        Returns:
            Frequency features, shape (batch_size, output_dim)
        """
        # Compute frequency transform
        amplitude, phase = self.compute_frequency_transform(x)

        # Stack amplitude and phase along channel dimension
        # (B, 3, H, W) + (B, 3, H, W) → (B, 6, H, W)
        freq_features = torch.cat([amplitude, phase], dim=1)

        # Pass through conv layers
        features = self.conv_layers(freq_features)

        # Adaptive pooling
        features = self.adaptive_pool(features)

        # Fully connected layers
        features = self.fc_layers(features)

        return features

    def visualize_frequency_spectrum(
        self, x: torch.Tensor, channel: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Visualize frequency spectrum for a single channel.

        Useful for debugging and understanding what the model sees.

        Args:
            x: Input image (1, C, H, W) or (C, H, W)
            channel: Which channel to visualize (0=R, 1=G, 2=B)

        Returns:
            amplitude, phase for visualization
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            amplitude, phase = self.compute_frequency_transform(x)

            # Extract specific channel
            amp_vis = amplitude[0, channel].cpu()
            phase_vis = phase[0, channel].cpu()

        return amp_vis, phase_vis
