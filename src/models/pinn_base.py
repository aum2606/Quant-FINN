"""
Base PINN Architecture for Financial Applications.

This module provides the foundational PINN architecture that other
models build upon. It includes:
- Flexible MLP backbone with smooth activations
- Input encoding options (raw, Fourier features)
- Residual connections for deep networks
- Interface for PDE residual computation

Key Design Choices:
------------------
1. Smooth activations (tanh, swish) - Required for second-order derivatives
2. Modular structure - Easy to extend with custom layers
3. Normalization-aware - Accounts for input/output scaling
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple, Callable
import math

from src.utils.autodiff import compute_gradient, compute_second_derivative


class SmoothActivation(nn.Module):
    """
    Collection of smooth activation functions suitable for PINNs.

    PINNs require computing second (or higher) order derivatives.
    ReLU and its variants have discontinuous derivatives, causing
    issues. These activations have smooth, well-defined derivatives
    at all points.
    """

    def __init__(self, activation_type: str = 'tanh'):
        """
        Initialize smooth activation.

        Args:
            activation_type: One of 'tanh', 'swish', 'gelu', 'softplus', 'sin'.
        """
        super().__init__()
        self.activation_type = activation_type

        if activation_type == 'tanh':
            self.fn = torch.tanh
        elif activation_type == 'swish':
            # Swish: x * sigmoid(x) - smooth, non-monotonic
            self.fn = lambda x: x * torch.sigmoid(x)
        elif activation_type == 'gelu':
            self.fn = nn.functional.gelu
        elif activation_type == 'softplus':
            # Softplus: log(1 + exp(x)) - smooth approximation to ReLU
            self.fn = nn.functional.softplus
        elif activation_type == 'sin':
            # Sine activation - perfect for periodic solutions
            self.fn = torch.sin
        else:
            raise ValueError(f"Unknown activation: {activation_type}")

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deep PINNs.

    For networks deeper than 4-5 layers, residual connections help:
    1. Gradient flow during backpropagation
    2. Training stability
    3. Feature reuse across layers

    Architecture:
        input -> Linear -> Activation -> Linear -> (+) -> Activation -> output
                                                    ^
                                                    |
                                                 (skip)
    """

    def __init__(
        self,
        hidden_dim: int,
        activation: str = 'tanh',
        dropout: float = 0.0,
    ):
        """
        Initialize residual block.

        Args:
            hidden_dim: Dimension of hidden layers.
            activation: Activation function type.
            dropout: Dropout probability (0 to disable).
        """
        super().__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = SmoothActivation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual block with skip connection."""
        residual = x

        # First transformation
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)

        # Second transformation
        out = self.linear2(out)

        # Skip connection and final activation
        out = self.activation(out + residual)

        return out


class PINNBase(nn.Module):
    """
    Base Physics-Informed Neural Network architecture.

    This provides a flexible MLP backbone suitable for solving PDEs.
    It handles:
    - Arbitrary input/output dimensions
    - Optional Fourier feature encoding
    - Residual connections for deep networks
    - Smooth activations for derivative computation

    Example:
        >>> pinn = PINNBase(
        ...     input_dim=2,    # (S, t)
        ...     output_dim=1,   # V
        ...     hidden_dim=64,
        ...     n_layers=4,
        ... )
        >>> V = pinn(S, t)

    Architecture:
        Input -> [Fourier Encoding] -> Linear -> [Residual Blocks] -> Linear -> Output
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 4,
        activation: str = 'tanh',
        use_fourier_encoding: bool = False,
        n_fourier_features: int = 10,
        fourier_scale: float = 1.0,
        use_residual: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initialize PINN base network.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output features.
            hidden_dim: Width of hidden layers.
            n_layers: Number of hidden layers.
            activation: Activation function ('tanh', 'swish', 'gelu').
            use_fourier_encoding: Whether to apply Fourier feature encoding.
            n_fourier_features: Number of Fourier frequency components.
            fourier_scale: Scale factor for Fourier features.
            use_residual: Whether to use residual connections.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_fourier_encoding = use_fourier_encoding

        # Determine effective input dimension after encoding
        if use_fourier_encoding:
            self.fourier_encoder = FourierEncoder(
                input_dim=input_dim,
                n_frequencies=n_fourier_features,
                scale=fourier_scale,
            )
            effective_input_dim = input_dim * 2 * n_fourier_features + input_dim
        else:
            self.fourier_encoder = None
            effective_input_dim = input_dim

        # Build network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(effective_input_dim, hidden_dim))
        layers.append(SmoothActivation(activation))

        # Hidden layers
        for _ in range(n_layers - 1):
            if use_residual and _ > 0:  # Use residual blocks after first hidden layer
                layers.append(ResidualBlock(hidden_dim, activation, dropout))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(SmoothActivation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        # Output layer (no activation - we want raw output)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Glorot initialization.

        This helps with gradient flow in deep networks and is particularly
        important for PINNs where we need stable derivative computation.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization works well with tanh
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, *inputs: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            *inputs: Variable number of input tensors to concatenate.
                     Each should have shape (batch_size,) or (batch_size, 1).

        Returns:
            Network output of shape (batch_size, output_dim).
        """
        # Stack inputs into single tensor
        x = self._prepare_inputs(*inputs)

        # Apply Fourier encoding if enabled
        if self.fourier_encoder is not None:
            x = self.fourier_encoder(x)

        # Forward through network
        return self.network(x)

    def _prepare_inputs(self, *inputs: Tensor) -> Tensor:
        """
        Prepare and concatenate input tensors.

        Handles various input shapes and stacks them appropriately.
        """
        processed = []
        for inp in inputs:
            if inp.dim() == 1:
                inp = inp.unsqueeze(-1)
            processed.append(inp)

        return torch.cat(processed, dim=-1)


class FourierEncoder(nn.Module):
    """
    Fourier Feature Encoding for inputs.

    Maps low-dimensional inputs to high-dimensional space using
    sinusoidal functions. This helps networks learn high-frequency
    patterns that they would otherwise miss (spectral bias).

    Transform: x -> [sin(2πf₁x), cos(2πf₁x), ..., sin(2πfₙx), cos(2πfₙx), x]

    Reference: "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains" (Tancik et al., 2020)
    """

    def __init__(
        self,
        input_dim: int,
        n_frequencies: int = 10,
        scale: float = 1.0,
        include_input: bool = True,
    ):
        """
        Initialize Fourier encoder.

        Args:
            input_dim: Dimension of input.
            n_frequencies: Number of frequency components.
            scale: Scale factor for frequencies.
            include_input: Whether to include raw input in output.
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_frequencies = n_frequencies
        self.include_input = include_input

        # Create frequency matrix (linear spacing)
        frequencies = scale * torch.arange(1, n_frequencies + 1, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Fourier encoding.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Encoded tensor of shape (batch, input_dim * 2 * n_freq [+ input_dim]).
        """
        # x: (batch, input_dim)
        # frequencies: (n_freq,)

        # Compute x * frequencies for all combinations
        # Result: (batch, input_dim, n_freq)
        x_proj = x.unsqueeze(-1) * self.frequencies * 2 * math.pi

        # Apply sin and cos
        sin_features = torch.sin(x_proj)
        cos_features = torch.cos(x_proj)

        # Concatenate: (batch, input_dim, 2 * n_freq)
        encoded = torch.cat([sin_features, cos_features], dim=-1)

        # Flatten: (batch, input_dim * 2 * n_freq)
        encoded = encoded.flatten(start_dim=1)

        # Optionally include raw input
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded

    @property
    def output_dim(self) -> int:
        """Get output dimension of encoder."""
        dim = self.input_dim * 2 * self.n_frequencies
        if self.include_input:
            dim += self.input_dim
        return dim


class BlackScholesPINN(PINNBase):
    """
    Specialized PINN for Black-Scholes option pricing.

    Extends PINNBase with Black-Scholes specific functionality:
    - PDE residual computation
    - Terminal and boundary condition handling
    - Greek computation

    The Black-Scholes PDE:
        ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

    Example:
        >>> pinn = BlackScholesPINN(
        ...     hidden_dim=64,
        ...     n_layers=4,
        ...     sigma=0.2,
        ...     r=0.05,
        ... )
        >>> V = pinn(S, t)
        >>> residual = pinn.pde_residual(S, t)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        activation: str = 'tanh',
        sigma: float = 0.2,
        r: float = 0.05,
        use_fourier_encoding: bool = True,
        n_fourier_features: int = 10,
        **kwargs,
    ):
        """
        Initialize Black-Scholes PINN.

        Args:
            hidden_dim: Width of hidden layers.
            n_layers: Number of hidden layers.
            activation: Activation function.
            sigma: Volatility (can be made learnable).
            r: Risk-free interest rate.
            use_fourier_encoding: Whether to use Fourier features.
            n_fourier_features: Number of Fourier frequencies.
            **kwargs: Additional arguments for PINNBase.
        """
        super().__init__(
            input_dim=2,  # (S, t)
            output_dim=1,  # V
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            use_fourier_encoding=use_fourier_encoding,
            n_fourier_features=n_fourier_features,
            **kwargs,
        )

        # PDE parameters (can be made learnable)
        self.register_buffer('sigma', torch.tensor(sigma))
        self.register_buffer('r', torch.tensor(r))

    def forward(self, S: Tensor, t: Tensor) -> Tensor:
        """
        Compute option value V(S, t).

        Args:
            S: Stock price tensor.
            t: Time tensor.

        Returns:
            Option value tensor.
        """
        return super().forward(S, t)

    def pde_residual(
        self,
        S: Tensor,
        t: Tensor,
        sigma: Optional[Tensor] = None,
        r: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Black-Scholes PDE residual.

        Residual = ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV

        This should be close to zero if the network satisfies the PDE.

        Args:
            S: Stock prices (must have requires_grad=True).
            t: Times (must have requires_grad=True).
            sigma: Optional volatility override.
            r: Optional rate override.

        Returns:
            PDE residual tensor.
        """
        sigma = sigma if sigma is not None else self.sigma
        r = r if r is not None else self.r

        # Ensure gradients are enabled
        S = S.requires_grad_(True) if not S.requires_grad else S
        t = t.requires_grad_(True) if not t.requires_grad else t

        # Forward pass
        V = self.forward(S, t)
        if V.dim() > 1:
            V = V.squeeze(-1)

        # Compute derivatives using autodiff
        dV_dS = compute_gradient(V, S)
        dV_dt = compute_gradient(V, t)
        d2V_dS2 = compute_second_derivative(V, S)

        # Black-Scholes PDE residual
        residual = (
            dV_dt
            + 0.5 * sigma**2 * S**2 * d2V_dS2
            + r * S * dV_dS
            - r * V
        )

        return residual

    def compute_greeks(
        self,
        S: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute option Greeks.

        Args:
            S: Stock prices.
            t: Times.

        Returns:
            Tuple of (delta, gamma, theta, vega*).
            *Note: Vega requires sigma to be a variable, not implemented here.
        """
        S = S.requires_grad_(True) if not S.requires_grad else S
        t = t.requires_grad_(True) if not t.requires_grad else t

        V = self.forward(S, t).squeeze(-1)

        delta = compute_gradient(V, S)
        gamma = compute_second_derivative(V, S)
        theta = compute_gradient(V, t)

        return delta, gamma, theta, V

    def terminal_condition_loss(
        self,
        S: Tensor,
        K: float,
        option_type: str = 'call',
    ) -> Tensor:
        """
        Compute loss for terminal condition V(S, T) = payoff(S).

        Args:
            S: Stock prices at terminal time.
            K: Strike price.
            option_type: 'call' or 'put'.

        Returns:
            MSE loss for terminal condition.
        """
        # At terminal time t = T (which we treat as t = 1 in normalized time)
        t = torch.ones_like(S)

        # Expected payoff
        if option_type == 'call':
            payoff = torch.relu(S - K)
        else:
            payoff = torch.relu(K - S)

        # Network prediction
        V = self.forward(S, t).squeeze(-1)

        return torch.nn.functional.mse_loss(V, payoff)

    def boundary_condition_loss(
        self,
        t: Tensor,
        S_min: float = 0.0,
        S_max: float = 200.0,
        K: float = 100.0,
        T: float = 1.0,
    ) -> Tensor:
        """
        Compute loss for boundary conditions.

        For a call option:
        - V(0, t) = 0 (worthless if stock is worthless)
        - V(S_max, t) ≈ S_max - K*exp(-r*(T-t)) (deep ITM)

        Args:
            t: Time values.
            S_min: Lower boundary (usually 0).
            S_max: Upper boundary (large S).
            K: Strike price.
            T: Expiration time.

        Returns:
            MSE loss for boundary conditions.
        """
        batch_size = t.shape[0]

        # Lower boundary: V(0, t) = 0
        S_lower = torch.zeros(batch_size, device=t.device)
        V_lower = self.forward(S_lower, t).squeeze(-1)
        loss_lower = V_lower.pow(2).mean()

        # Upper boundary: V(S_max, t) ≈ S_max - K*exp(-r*(T-t))
        S_upper = torch.full((batch_size,), S_max, device=t.device)
        V_upper = self.forward(S_upper, t).squeeze(-1)
        expected_upper = S_max - K * torch.exp(-self.r * (T - t))
        loss_upper = (V_upper - expected_upper).pow(2).mean()

        return loss_lower + loss_upper
