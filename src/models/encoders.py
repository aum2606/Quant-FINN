"""
Feature Encoders for Quant-FINN.

This module provides encoding networks that transform raw inputs
into meaningful representations:
- FourierFeatureEncoder: Overcomes spectral bias with frequency encoding
- TemporalEncoder: Captures sequential patterns in time series
- FeatureFusion: Combines multiple encoded features

These encoders are the first stage of the Quant-FINN pipeline,
transforming diverse inputs (prices, technicals, sentiment) into
a unified representation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple
import math


class FourierFeatureEncoder(nn.Module):
    """
    Fourier Feature Encoder with learnable frequencies.

    Extends basic Fourier encoding by:
    1. Using learnable (or fixed) frequency matrix
    2. Optional random Fourier features for better coverage
    3. Projection to desired output dimension

    This is critical for PINNs because neural networks have "spectral bias" -
    they naturally learn low-frequency functions first. Fourier features
    help capture high-frequency components of the solution.

    Reference: Tancik et al. "Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains" (NeurIPS 2020)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        n_frequencies: int = 10,
        scale: float = 1.0,
        learnable: bool = False,
        include_input: bool = True,
    ):
        """
        Initialize Fourier feature encoder.

        Args:
            input_dim: Dimension of input features.
            output_dim: Desired output dimension (after projection).
            n_frequencies: Number of Fourier frequency components.
            scale: Scale factor for frequencies.
            learnable: Whether frequencies should be learned.
            include_input: Whether to concatenate raw input.
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_frequencies = n_frequencies
        self.include_input = include_input

        # Frequency matrix: (input_dim, n_frequencies)
        # Initialize with linear spacing
        freq_init = scale * torch.linspace(1, n_frequencies, n_frequencies)
        freq_matrix = freq_init.unsqueeze(0).expand(input_dim, -1)

        if learnable:
            self.frequencies = nn.Parameter(freq_matrix.clone())
        else:
            self.register_buffer('frequencies', freq_matrix)

        # Compute intermediate dimension after Fourier transform
        fourier_dim = input_dim * 2 * n_frequencies
        if include_input:
            fourier_dim += input_dim

        # Projection to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Fourier encoding and projection.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Encoded tensor of shape (batch, output_dim).
        """
        batch_size = x.shape[0]

        # x: (batch, input_dim)
        # frequencies: (input_dim, n_freq)
        # We want: (batch, input_dim, n_freq)

        # Compute x_i * f_j for all combinations
        x_proj = x.unsqueeze(-1) * self.frequencies.unsqueeze(0) * 2 * math.pi

        # Apply sin and cos
        sin_features = torch.sin(x_proj)  # (batch, input_dim, n_freq)
        cos_features = torch.cos(x_proj)  # (batch, input_dim, n_freq)

        # Concatenate and flatten
        encoded = torch.cat([sin_features, cos_features], dim=-1)
        encoded = encoded.view(batch_size, -1)

        # Optionally include raw input
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        # Project to output dimension
        return self.projection(encoded)


class TemporalEncoder(nn.Module):
    """
    Temporal Encoder using LSTM for sequential financial data.

    Processes time series data (price history, returns, etc.) to extract
    temporal patterns. Outputs a fixed-size representation suitable for
    the physics-informed layers.

    Architecture:
        Input sequence -> LSTM -> Hidden state -> Linear -> Encoded features

    This encoder captures:
    - Trend information
    - Momentum patterns
    - Volatility clustering (to some extent)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Initialize temporal encoder.

        Args:
            input_dim: Number of features per timestep (e.g., 1 for returns).
            hidden_dim: LSTM hidden dimension.
            output_dim: Output feature dimension.
            n_layers: Number of LSTM layers.
            dropout: Dropout probability between LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output dimension of LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Project to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode temporal sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            lengths: Optional lengths for variable-length sequences.

        Returns:
            Encoded tensor of shape (batch, output_dim).
        """
        # Pack sequence if variable length
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM forward pass
        _, (h_n, _) = self.lstm(x)

        # h_n shape: (n_layers * num_directions, batch, hidden_dim)
        # Take the last layer's hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_final = h_n[-1]

        # Project to output dimension
        return self.projection(h_final)


class ConvTemporalEncoder(nn.Module):
    """
    Convolutional Temporal Encoder for time series.

    Uses 1D convolutions to capture local patterns, often more efficient
    than LSTM for long sequences. Good for capturing:
    - Short-term price patterns
    - Technical indicator patterns
    - Multi-scale features with dilated convolutions
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 64,
        channels: List[int] = [32, 64, 64],
        kernel_sizes: List[int] = [5, 5, 3],
        pool_size: int = 2,
    ):
        """
        Initialize convolutional encoder.

        Args:
            input_dim: Number of input channels.
            output_dim: Output feature dimension.
            channels: Number of channels in each conv layer.
            kernel_sizes: Kernel size for each conv layer.
            pool_size: Pooling size between conv layers.
        """
        super().__init__()

        layers = []
        in_channels = input_dim

        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),  # Can use ReLU here since we're not differentiating through this
                nn.MaxPool1d(pool_size),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Adaptive pooling to handle variable sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], output_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode temporal sequence using convolutions.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Encoded tensor of shape (batch, output_dim).
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Apply conv layers
        x = self.conv_layers(x)

        # Global pooling
        x = self.adaptive_pool(x).squeeze(-1)

        # Project
        return self.projection(x)


class FeatureFusion(nn.Module):
    """
    Feature Fusion module combining multiple encoded features.

    Combines outputs from different encoders (temporal, Fourier, etc.)
    using attention or simple concatenation. This creates a unified
    representation for the physics-informed core.

    Fusion strategies:
    - 'concat': Simple concatenation + linear projection
    - 'attention': Multi-head attention fusion
    - 'gated': Gated fusion with learned importance weights
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dim: int,
        fusion_type: str = 'concat',
        n_heads: int = 4,
    ):
        """
        Initialize feature fusion module.

        Args:
            input_dims: List of input dimensions from each encoder.
            output_dim: Desired output dimension.
            fusion_type: 'concat', 'attention', or 'gated'.
            n_heads: Number of attention heads (if using attention).
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.input_dims = input_dims
        total_input_dim = sum(input_dims)

        if fusion_type == 'concat':
            # Simple concatenation with projection
            self.fusion = nn.Sequential(
                nn.Linear(total_input_dim, output_dim),
                nn.Tanh(),
            )

        elif fusion_type == 'attention':
            # Attention-based fusion
            # First project all inputs to same dimension
            self.projections = nn.ModuleList([
                nn.Linear(dim, output_dim) for dim in input_dims
            ])
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=n_heads,
                batch_first=True,
            )
            self.output_proj = nn.Linear(output_dim, output_dim)

        elif fusion_type == 'gated':
            # Gated fusion
            self.projections = nn.ModuleList([
                nn.Linear(dim, output_dim) for dim in input_dims
            ])
            # Gate network learns importance of each input
            self.gate = nn.Sequential(
                nn.Linear(total_input_dim, len(input_dims)),
                nn.Softmax(dim=-1),
            )

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, *features: Tensor) -> Tensor:
        """
        Fuse multiple feature tensors.

        Args:
            *features: Variable number of feature tensors.
                       Each has shape (batch, feature_dim_i).

        Returns:
            Fused tensor of shape (batch, output_dim).
        """
        if self.fusion_type == 'concat':
            # Concatenate and project
            concatenated = torch.cat(features, dim=-1)
            return self.fusion(concatenated)

        elif self.fusion_type == 'attention':
            # Project to common dimension
            projected = [proj(f).unsqueeze(1) for proj, f in zip(self.projections, features)]
            # Stack: (batch, n_features, output_dim)
            stacked = torch.cat(projected, dim=1)

            # Self-attention
            attended, _ = self.attention(stacked, stacked, stacked)

            # Pool over features
            pooled = attended.mean(dim=1)

            return self.output_proj(pooled)

        elif self.fusion_type == 'gated':
            # Compute gate weights
            concatenated = torch.cat(features, dim=-1)
            gates = self.gate(concatenated)  # (batch, n_features)

            # Project each feature
            projected = [proj(f) for proj, f in zip(self.projections, features)]

            # Weighted sum
            fused = sum(g.unsqueeze(-1) * p for g, p in zip(gates.unbind(dim=-1), projected))

            return fused


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for sequence models.

    Adds position information to embeddings using sinusoidal functions.
    This is the standard positional encoding from "Attention Is All You Need".

    Useful when using Transformers for time series processing.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension (embedding size).
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
