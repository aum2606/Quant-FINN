"""
PDE Residual Computations for Financial PDEs.

This module provides residual functions for common financial PDEs:
1. Black-Scholes: European option pricing
2. Heston: Stochastic volatility model
3. General diffusion: Custom SDE-based PDEs

The residual R should be ~0 if the network satisfies the PDE.
The PDE loss is typically L_PDE = ||R||² averaged over collocation points.

Key Implementation Details:
- All computations use autodiff for derivatives
- create_graph=True is essential for backprop through residuals
- Inputs must have requires_grad=True before forward pass
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional

from src.utils.autodiff import (
    compute_gradient,
    compute_second_derivative,
    compute_mixed_derivative,
)


def compute_black_scholes_residual(
    network: nn.Module,
    S: Tensor,
    t: Tensor,
    sigma: float,
    r: float,
) -> Tensor:
    """
    Compute Black-Scholes PDE residual.

    The Black-Scholes PDE:
        ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

    This function computes the left-hand side (residual), which should
    be zero for points that satisfy the PDE.

    Args:
        network: Neural network V(S, t).
        S: Stock prices (must have requires_grad=True).
        t: Times (must have requires_grad=True).
        sigma: Volatility (constant).
        r: Risk-free rate.

    Returns:
        Residual tensor (same shape as S).

    Example:
        >>> S = torch.linspace(50, 150, 100).requires_grad_(True)
        >>> t = torch.rand(100).requires_grad_(True)
        >>> residual = compute_black_scholes_residual(pinn, S, t, 0.2, 0.05)
        >>> pde_loss = residual.pow(2).mean()
    """
    # Ensure gradient tracking
    S = S.requires_grad_(True) if not S.requires_grad else S
    t = t.requires_grad_(True) if not t.requires_grad else t

    # Forward pass
    V = network(S, t)
    if V.dim() > 1:
        V = V.squeeze(-1)

    # First derivatives
    dV_dS = compute_gradient(V, S)
    dV_dt = compute_gradient(V, t)

    # Second derivative (Gamma)
    d2V_dS2 = compute_second_derivative(V, S)

    # Black-Scholes residual
    residual = (
        dV_dt
        + 0.5 * sigma**2 * S**2 * d2V_dS2
        + r * S * dV_dS
        - r * V
    )

    return residual


class BlackScholesResidual(nn.Module):
    """
    Black-Scholes PDE Residual as a Module.

    Encapsulates the Black-Scholes residual computation with
    configurable parameters. Can handle:
    - Constant volatility
    - Time-varying volatility (term structure)
    - Asset-dependent volatility (local vol)

    The PDE (in backward time τ = T - t):
        ∂V/∂τ = ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV

    Note: We use forward time t in implementation, so signs differ slightly.
    """

    def __init__(
        self,
        sigma: float = 0.2,
        r: float = 0.05,
        sigma_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
        """
        Initialize Black-Scholes residual module.

        Args:
            sigma: Constant volatility (used if sigma_fn is None).
            r: Risk-free interest rate.
            sigma_fn: Optional function sigma(S, t) for local volatility.
        """
        super().__init__()

        self.register_buffer('_sigma', torch.tensor(sigma))
        self.register_buffer('_r', torch.tensor(r))
        self.sigma_fn = sigma_fn

    def forward(
        self,
        network: nn.Module,
        S: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Compute Black-Scholes residual.

        Args:
            network: Price network V(S, t).
            S: Stock prices.
            t: Times.

        Returns:
            PDE residual tensor.
        """
        # Get volatility
        if self.sigma_fn is not None:
            sigma = self.sigma_fn(S, t)
        else:
            sigma = self._sigma

        return compute_black_scholes_residual(network, S, t, sigma, self._r)

    def loss(
        self,
        network: nn.Module,
        S: Tensor,
        t: Tensor,
        reduction: str = 'mean',
    ) -> Tensor:
        """
        Compute PDE loss (squared residual).

        Args:
            network: Price network.
            S: Stock prices.
            t: Times.
            reduction: 'mean', 'sum', or 'none'.

        Returns:
            Scalar loss value.
        """
        residual = self.forward(network, S, t)
        loss = residual.pow(2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HestonResidual(nn.Module):
    """
    Heston Stochastic Volatility PDE Residual.

    The Heston model has two state variables: S (price) and v (variance).

    SDEs:
        dS = μS dt + √v S dW₁
        dv = κ(θ - v) dt + ξ√v dW₂
        dW₁ · dW₂ = ρ dt

    Pricing PDE:
        ∂V/∂t + ½vS²(∂²V/∂S²) + ρξvS(∂²V/∂S∂v) + ½ξ²v(∂²V/∂v²)
              + rS(∂V/∂S) + [κ(θ-v) - λv](∂V/∂v) - rV = 0

    Parameters:
        κ (kappa): Mean reversion speed of variance
        θ (theta): Long-term variance level
        ξ (xi): Volatility of variance (vol-of-vol)
        ρ (rho): Correlation between price and variance
        λ (lambda): Market price of volatility risk

    Note: This is a 2D PDE, more challenging than Black-Scholes.
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        r: float = 0.05,
        lambda_v: float = 0.0,
    ):
        """
        Initialize Heston residual module.

        Args:
            kappa: Mean reversion speed.
            theta: Long-term variance.
            xi: Volatility of variance.
            rho: Correlation.
            r: Risk-free rate.
            lambda_v: Variance risk premium.
        """
        super().__init__()

        self.register_buffer('kappa', torch.tensor(kappa))
        self.register_buffer('theta', torch.tensor(theta))
        self.register_buffer('xi', torch.tensor(xi))
        self.register_buffer('rho', torch.tensor(rho))
        self.register_buffer('r', torch.tensor(r))
        self.register_buffer('lambda_v', torch.tensor(lambda_v))

    def forward(
        self,
        network: nn.Module,
        S: Tensor,
        v: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Compute Heston PDE residual.

        Args:
            network: Price network V(S, v, t).
            S: Stock prices.
            v: Variance values.
            t: Times.

        Returns:
            PDE residual tensor.
        """
        # Ensure gradient tracking
        S = S.requires_grad_(True) if not S.requires_grad else S
        v = v.requires_grad_(True) if not v.requires_grad else v
        t = t.requires_grad_(True) if not t.requires_grad else t

        # Forward pass
        V = network(S, v, t)
        if V.dim() > 1:
            V = V.squeeze(-1)

        # First derivatives
        dV_dS = compute_gradient(V, S)
        dV_dv = compute_gradient(V, v)
        dV_dt = compute_gradient(V, t)

        # Second derivatives
        d2V_dS2 = compute_second_derivative(V, S)
        d2V_dv2 = compute_second_derivative(V, v)

        # Mixed derivative
        d2V_dSdv = compute_mixed_derivative(V, S, v)

        # Heston PDE residual
        residual = (
            dV_dt
            + 0.5 * v * S**2 * d2V_dS2
            + self.rho * self.xi * v * S * d2V_dSdv
            + 0.5 * self.xi**2 * v * d2V_dv2
            + self.r * S * dV_dS
            + (self.kappa * (self.theta - v) - self.lambda_v * v) * dV_dv
            - self.r * V
        )

        return residual

    def loss(
        self,
        network: nn.Module,
        S: Tensor,
        v: Tensor,
        t: Tensor,
        reduction: str = 'mean',
    ) -> Tensor:
        """Compute Heston PDE loss."""
        residual = self.forward(network, S, v, t)
        loss = residual.pow(2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GeneralDiffusionResidual(nn.Module):
    """
    General Diffusion PDE Residual.

    For a general diffusion process:
        dX = μ(X, t) dt + σ(X, t) dW

    The Feynman-Kac formula gives the PDE for pricing:
        ∂V/∂t + μ(∂V/∂X) + ½σ²(∂²V/∂X²) - rV = 0

    This class allows specifying custom drift and diffusion functions.
    """

    def __init__(
        self,
        drift_fn: Callable[[Tensor, Tensor], Tensor],
        diffusion_fn: Callable[[Tensor, Tensor], Tensor],
        r: float = 0.05,
    ):
        """
        Initialize general diffusion residual.

        Args:
            drift_fn: Function μ(X, t) returning drift coefficient.
            diffusion_fn: Function σ(X, t) returning diffusion coefficient.
            r: Discount rate.
        """
        super().__init__()

        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.register_buffer('r', torch.tensor(r))

    def forward(
        self,
        network: nn.Module,
        X: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Compute general diffusion PDE residual.

        Args:
            network: Value network V(X, t).
            X: State variable values.
            t: Times.

        Returns:
            PDE residual tensor.
        """
        X = X.requires_grad_(True) if not X.requires_grad else X
        t = t.requires_grad_(True) if not t.requires_grad else t

        V = network(X, t)
        if V.dim() > 1:
            V = V.squeeze(-1)

        # Get drift and diffusion
        mu = self.drift_fn(X, t)
        sigma = self.diffusion_fn(X, t)

        # Derivatives
        dV_dX = compute_gradient(V, X)
        dV_dt = compute_gradient(V, t)
        d2V_dX2 = compute_second_derivative(V, X)

        # Feynman-Kac PDE residual
        residual = (
            dV_dt
            + mu * dV_dX
            + 0.5 * sigma**2 * d2V_dX2
            - self.r * V
        )

        return residual


class OrnsteinUhlenbeckResidual(nn.Module):
    """
    Ornstein-Uhlenbeck Process Residual.

    The OU process is commonly used for modeling mean-reverting quantities
    like interest rates and volatility:
        dX = κ(θ - X) dt + σ dW

    where:
        κ: Mean reversion speed
        θ: Long-term mean
        σ: Volatility
    """

    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.0,
        sigma: float = 0.1,
        r: float = 0.05,
    ):
        """
        Initialize OU residual module.

        Args:
            kappa: Mean reversion speed.
            theta: Long-term mean.
            sigma: Volatility.
            r: Discount rate.
        """
        super().__init__()

        self.register_buffer('kappa', torch.tensor(kappa))
        self.register_buffer('theta', torch.tensor(theta))
        self.register_buffer('sigma', torch.tensor(sigma))
        self.register_buffer('r', torch.tensor(r))

    def forward(
        self,
        network: nn.Module,
        X: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute OU PDE residual."""
        X = X.requires_grad_(True) if not X.requires_grad else X
        t = t.requires_grad_(True) if not t.requires_grad else t

        V = network(X, t)
        if V.dim() > 1:
            V = V.squeeze(-1)

        dV_dX = compute_gradient(V, X)
        dV_dt = compute_gradient(V, t)
        d2V_dX2 = compute_second_derivative(V, X)

        # OU drift: κ(θ - X)
        drift = self.kappa * (self.theta - X)

        residual = (
            dV_dt
            + drift * dV_dX
            + 0.5 * self.sigma**2 * d2V_dX2
            - self.r * V
        )

        return residual
