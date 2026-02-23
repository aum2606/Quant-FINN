"""
Normalization Utilities for Financial Data.

Proper normalization is critical for PINN training:
1. Improves convergence by keeping values in reasonable ranges
2. Prevents numerical instability in derivative computations
3. Must be accounted for when computing PDE residuals

IMPORTANT: When computing derivatives with normalized data, the chain rule
must be applied. For example:
    If V_norm = (V - μ_V) / σ_V and S_norm = (S - μ_S) / σ_S
    Then: dV/dS = (σ_V / σ_S) * dV_norm/dS_norm
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Any
import numpy as np


class Normalizer:
    """
    Standard normalization using mean and standard deviation.

    Transforms data to have zero mean and unit variance:
        x_norm = (x - mean) / std

    This is the default choice for most financial data where values
    can be both positive and negative (e.g., returns).

    Attributes:
        mean: Mean of the training data.
        std: Standard deviation of the training data.
        eps: Small constant to prevent division by zero.

    Example:
        >>> normalizer = Normalizer()
        >>> normalizer.fit(train_returns)
        >>> normalized = normalizer.normalize(returns)
        >>> original = normalizer.denormalize(normalized)
    """

    def __init__(self, eps: float = 1e-8):
        """
        Initialize normalizer.

        Args:
            eps: Small constant added to std to prevent division by zero.
        """
        self.eps = eps
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None
        self._fitted = False

    def fit(self, data: Tensor, dim: int = 0) -> "Normalizer":
        """
        Compute normalization statistics from data.

        Args:
            data: Training data tensor.
            dim: Dimension along which to compute statistics.
                 Default is 0 (compute stats over batch dimension).

        Returns:
            self (for method chaining).
        """
        self.mean = data.mean(dim=dim)
        self.std = data.std(dim=dim) + self.eps
        self._fitted = True
        return self

    def normalize(self, data: Tensor) -> Tensor:
        """
        Apply normalization to data.

        Args:
            data: Data tensor to normalize.

        Returns:
            Normalized data with zero mean and unit variance.

        Raises:
            RuntimeError: If fit() hasn't been called.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before use. Call fit() first.")
        return (data - self.mean) / self.std

    def denormalize(self, data: Tensor) -> Tensor:
        """
        Reverse normalization to get original scale.

        Args:
            data: Normalized data tensor.

        Returns:
            Data in original scale.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before use. Call fit() first.")
        return data * self.std + self.mean

    def derivative_scale(self, input_idx: int = 0, output_idx: int = 0) -> Tensor:
        """
        Get the scaling factor for derivatives.

        When computing ∂y/∂x with normalized data:
            ∂y/∂x = (σ_y / σ_x) * ∂y_norm/∂x_norm

        Args:
            input_idx: Index of the input dimension.
            output_idx: Index of the output dimension.

        Returns:
            Scaling factor σ_y / σ_x for the derivative.
        """
        # Get the appropriate std values
        if self.std.dim() == 0:
            return torch.ones(1, device=self.std.device)

        std_y = self.std[output_idx] if output_idx < len(self.std) else self.std[-1]
        std_x = self.std[input_idx] if input_idx < len(self.std) else self.std[-1]

        return std_y / std_x

    def state_dict(self) -> Dict[str, Any]:
        """Save normalizer state for serialization."""
        return {
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load normalizer state from dict."""
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state["eps"]
        self._fitted = state["fitted"]


class LogNormalizer:
    """
    Log-transform normalization for positive-valued data.

    Applies log transformation before standard normalization:
        x_norm = (log(x) - mean_log) / std_log

    This is appropriate for data that must be positive and spans
    multiple orders of magnitude, such as:
    - Stock prices
    - Option values
    - Volatility (variance)

    The log transform also converts multiplicative relationships to
    additive ones, which can help with learning.

    Example:
        >>> normalizer = LogNormalizer()
        >>> normalizer.fit(prices)
        >>> log_normalized = normalizer.normalize(prices)
        >>> original = normalizer.denormalize(log_normalized)
    """

    def __init__(self, eps: float = 1e-8):
        """
        Initialize log normalizer.

        Args:
            eps: Small constant to prevent log(0).
        """
        self.eps = eps
        self.log_mean: Optional[Tensor] = None
        self.log_std: Optional[Tensor] = None
        self._fitted = False

    def fit(self, data: Tensor, dim: int = 0) -> "LogNormalizer":
        """
        Compute normalization statistics from log-transformed data.

        Args:
            data: Training data tensor (must be positive).
            dim: Dimension along which to compute statistics.

        Returns:
            self (for method chaining).
        """
        # Apply log transform
        log_data = torch.log(data + self.eps)

        # Compute statistics on log-transformed data
        self.log_mean = log_data.mean(dim=dim)
        self.log_std = log_data.std(dim=dim) + self.eps
        self._fitted = True

        return self

    def normalize(self, data: Tensor) -> Tensor:
        """
        Apply log-normalization to data.

        Args:
            data: Positive-valued data tensor.

        Returns:
            Log-normalized data.
        """
        if not self._fitted:
            raise RuntimeError("LogNormalizer must be fitted before use.")

        log_data = torch.log(data + self.eps)
        return (log_data - self.log_mean) / self.log_std

    def denormalize(self, data: Tensor) -> Tensor:
        """
        Reverse log-normalization to get original scale.

        Args:
            data: Log-normalized data tensor.

        Returns:
            Data in original scale (positive values).
        """
        if not self._fitted:
            raise RuntimeError("LogNormalizer must be fitted before use.")

        log_data = data * self.log_std + self.log_mean
        return torch.exp(log_data) - self.eps


class FeatureNormalizer:
    """
    Multi-feature normalizer with per-feature statistics.

    Handles multiple input features with potentially different
    normalization strategies. Useful for PINN inputs where:
    - Stock price might use log normalization
    - Time might use standard normalization
    - Volatility might use log normalization

    Example:
        >>> normalizer = FeatureNormalizer({
        ...     'S': 'log',      # Stock price
        ...     't': 'standard', # Time
        ...     'sigma': 'log',  # Volatility
        ...     'r': 'standard', # Interest rate
        ... })
        >>> normalizer.fit(data_dict)
    """

    def __init__(
        self,
        feature_types: Dict[str, str],
        eps: float = 1e-8
    ):
        """
        Initialize multi-feature normalizer.

        Args:
            feature_types: Dict mapping feature names to normalization type.
                          Types: 'standard', 'log', 'none'.
            eps: Small constant for numerical stability.
        """
        self.feature_types = feature_types
        self.eps = eps
        self.normalizers: Dict[str, Normalizer] = {}

        # Create normalizer for each feature
        for name, norm_type in feature_types.items():
            if norm_type == 'log':
                self.normalizers[name] = LogNormalizer(eps=eps)
            elif norm_type == 'standard':
                self.normalizers[name] = Normalizer(eps=eps)
            elif norm_type == 'none':
                self.normalizers[name] = None
            else:
                raise ValueError(f"Unknown normalization type: {norm_type}")

    def fit(self, data: Dict[str, Tensor]) -> "FeatureNormalizer":
        """
        Fit all normalizers to their respective features.

        Args:
            data: Dict mapping feature names to data tensors.

        Returns:
            self (for method chaining).
        """
        for name, normalizer in self.normalizers.items():
            if normalizer is not None and name in data:
                normalizer.fit(data[name])
        return self

    def normalize(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Normalize all features.

        Args:
            data: Dict of feature tensors.

        Returns:
            Dict of normalized feature tensors.
        """
        result = {}
        for name, tensor in data.items():
            if name in self.normalizers and self.normalizers[name] is not None:
                result[name] = self.normalizers[name].normalize(tensor)
            else:
                result[name] = tensor
        return result

    def denormalize(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Denormalize all features.

        Args:
            data: Dict of normalized feature tensors.

        Returns:
            Dict of denormalized feature tensors.
        """
        result = {}
        for name, tensor in data.items():
            if name in self.normalizers and self.normalizers[name] is not None:
                result[name] = self.normalizers[name].denormalize(tensor)
            else:
                result[name] = tensor
        return result
