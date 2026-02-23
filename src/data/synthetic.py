"""
Synthetic Data Generation for PINN Testing.

This module generates synthetic financial data following
known stochastic processes:
- Geometric Brownian Motion (GBM)
- Heston stochastic volatility
- Black-Scholes option prices

Synthetic data is useful for:
- Validating PINN implementations
- Unit testing
- Demonstrations
- Benchmarking against analytical solutions
"""

import torch
from torch import Tensor
from typing import Tuple, Optional, Dict
import numpy as np
from scipy.stats import norm


def generate_gbm_paths(
    S0: float = 100.0,
    mu: float = 0.1,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 1000,
    seed: Optional[int] = None,
    device: str = 'cpu',
) -> Tuple[Tensor, Tensor]:
    """
    Generate stock price paths following Geometric Brownian Motion.

    dS = μS dt + σS dW

    The analytical solution is:
        S_t = S_0 * exp((μ - σ²/2)t + σW_t)

    Args:
        S0: Initial stock price.
        mu: Drift (expected return).
        sigma: Volatility.
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of paths to simulate.
        seed: Random seed for reproducibility.
        device: Device for tensors.

    Returns:
        Tuple of (times, paths) where:
            - times: (n_steps + 1,) time points
            - paths: (n_paths, n_steps + 1) price paths
    """
    if seed is not None:
        torch.manual_seed(seed)

    dt = T / n_steps
    times = torch.linspace(0, T, n_steps + 1, device=device)

    # Generate random increments
    dW = torch.randn(n_paths, n_steps, device=device) * np.sqrt(dt)

    # Cumulative sum for Brownian motion
    W = torch.cumsum(dW, dim=1)
    W = torch.cat([torch.zeros(n_paths, 1, device=device), W], dim=1)

    # GBM formula
    paths = S0 * torch.exp((mu - 0.5 * sigma**2) * times + sigma * W)

    return times, paths


def generate_heston_paths(
    S0: float = 100.0,
    v0: float = 0.04,
    mu: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    T: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 1000,
    seed: Optional[int] = None,
    device: str = 'cpu',
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Generate paths following the Heston stochastic volatility model.

    dS = μS dt + √v S dW₁
    dv = κ(θ - v) dt + ξ√v dW₂
    dW₁ · dW₂ = ρ dt

    Args:
        S0: Initial stock price.
        v0: Initial variance.
        mu: Drift.
        kappa: Mean reversion speed.
        theta: Long-term variance.
        xi: Volatility of volatility.
        rho: Correlation between price and variance.
        T: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of paths.
        seed: Random seed.
        device: Device for tensors.

    Returns:
        Tuple of (times, price_paths, variance_paths).
    """
    if seed is not None:
        torch.manual_seed(seed)

    dt = T / n_steps
    times = torch.linspace(0, T, n_steps + 1, device=device)

    # Generate correlated Brownian increments
    Z1 = torch.randn(n_paths, n_steps, device=device)
    Z2 = torch.randn(n_paths, n_steps, device=device)

    dW1 = Z1 * np.sqrt(dt)
    dW2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

    # Initialize paths
    S = torch.zeros(n_paths, n_steps + 1, device=device)
    v = torch.zeros(n_paths, n_steps + 1, device=device)
    S[:, 0] = S0
    v[:, 0] = v0

    # Euler-Maruyama discretization
    for t in range(n_steps):
        # Ensure variance stays positive (truncation scheme)
        v_pos = torch.clamp(v[:, t], min=1e-8)
        sqrt_v = torch.sqrt(v_pos)

        # Price update
        S[:, t + 1] = S[:, t] * (1 + mu * dt + sqrt_v * dW1[:, t])

        # Variance update (with reflection at zero)
        v[:, t + 1] = v[:, t] + kappa * (theta - v_pos) * dt + xi * sqrt_v * dW2[:, t]
        v[:, t + 1] = torch.abs(v[:, t + 1])  # Reflection scheme

    return times, S, v


def generate_black_scholes_data(
    S_range: Tuple[float, float] = (50.0, 150.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    n_points: int = 1000,
    option_type: str = 'call',
    add_noise: bool = False,
    noise_level: float = 0.01,
    seed: Optional[int] = None,
    device: str = 'cpu',
) -> Dict[str, Tensor]:
    """
    Generate Black-Scholes option prices (analytical solution).

    Useful for validating PINN implementations - the network
    should be able to learn these prices exactly.

    Args:
        S_range: (min, max) stock price range.
        t_range: (min, max) time range (time to expiration).
        K: Strike price.
        r: Risk-free rate.
        sigma: Volatility.
        n_points: Number of points to generate.
        option_type: 'call' or 'put'.
        add_noise: Whether to add noise to prices.
        noise_level: Standard deviation of noise (as fraction of price).
        seed: Random seed.
        device: Device for tensors.

    Returns:
        Dict with 'S', 't', 'V' (option price), 'delta', 'gamma'.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate random points
    S = torch.rand(n_points, device=device) * (S_range[1] - S_range[0]) + S_range[0]
    t = torch.rand(n_points, device=device) * (t_range[1] - t_range[0]) + t_range[0]

    # Time to expiration (assuming t_range[1] is expiration)
    tau = t_range[1] - t

    # Black-Scholes formula
    S_np = S.cpu().numpy()
    tau_np = tau.cpu().numpy()

    # Handle tau = 0 case
    tau_np = np.maximum(tau_np, 1e-8)

    d1 = (np.log(S_np / K) + (r + 0.5 * sigma**2) * tau_np) / (sigma * np.sqrt(tau_np))
    d2 = d1 - sigma * np.sqrt(tau_np)

    if option_type == 'call':
        price = S_np * norm.cdf(d1) - K * np.exp(-r * tau_np) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * tau_np) * norm.cdf(-d2) - S_np * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (S_np * sigma * np.sqrt(tau_np) + 1e-8)

    # Convert to tensors
    V = torch.tensor(price, dtype=torch.float32, device=device)
    delta_tensor = torch.tensor(delta, dtype=torch.float32, device=device)
    gamma_tensor = torch.tensor(gamma, dtype=torch.float32, device=device)

    # Add noise if requested
    if add_noise:
        noise = torch.randn_like(V) * noise_level * V.abs()
        V = V + noise

    return {
        'S': S,
        't': t,
        'V': V,
        'delta': delta_tensor,
        'gamma': gamma_tensor,
        'tau': torch.tensor(tau_np, dtype=torch.float32, device=device),
    }


def generate_garch_data(
    n_samples: int = 1000,
    omega: float = 0.0001,
    alpha: float = 0.1,
    beta: float = 0.85,
    seed: Optional[int] = None,
    device: str = 'cpu',
) -> Dict[str, Tensor]:
    """
    Generate returns following GARCH(1,1) dynamics.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    r_t = σ_t · z_t, z_t ~ N(0,1)

    Args:
        n_samples: Number of time steps.
        omega: Long-run variance weight.
        alpha: ARCH coefficient.
        beta: GARCH coefficient.
        seed: Random seed.
        device: Device for tensors.

    Returns:
        Dict with 'returns', 'variance', 'volatility'.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize
    variance = np.zeros(n_samples)
    returns = np.zeros(n_samples)

    # Unconditional variance as starting point
    uncond_var = omega / (1 - alpha - beta)
    variance[0] = uncond_var

    # Generate innovations
    z = np.random.randn(n_samples)

    for t in range(n_samples):
        # Return
        returns[t] = np.sqrt(variance[t]) * z[t]

        # Update variance for next period
        if t < n_samples - 1:
            variance[t + 1] = omega + alpha * returns[t]**2 + beta * variance[t]

    return {
        'returns': torch.tensor(returns, dtype=torch.float32, device=device),
        'variance': torch.tensor(variance, dtype=torch.float32, device=device),
        'volatility': torch.tensor(np.sqrt(variance), dtype=torch.float32, device=device),
    }


def generate_factor_data(
    n_samples: int = 1000,
    n_assets: int = 10,
    n_factors: int = 3,
    factor_volatilities: Optional[list] = None,
    idio_volatility: float = 0.1,
    seed: Optional[int] = None,
    device: str = 'cpu',
) -> Dict[str, Tensor]:
    """
    Generate asset returns following a factor model.

    R_i = α_i + Σ_j β_{ij} F_j + ε_i

    Args:
        n_samples: Number of time periods.
        n_assets: Number of assets.
        n_factors: Number of factors.
        factor_volatilities: Volatility of each factor.
        idio_volatility: Idiosyncratic volatility.
        seed: Random seed.
        device: Device for tensors.

    Returns:
        Dict with 'asset_returns', 'factor_returns', 'betas', 'alphas'.
    """
    if seed is not None:
        np.random.seed(seed)

    if factor_volatilities is None:
        factor_volatilities = [0.15, 0.1, 0.08][:n_factors]

    # Generate betas (factor loadings)
    betas = np.random.randn(n_assets, n_factors) * 0.5
    betas[:, 0] = np.abs(betas[:, 0]) + 0.5  # Market beta positive

    # Generate alphas (small, centered at zero)
    alphas = np.random.randn(n_assets) * 0.001

    # Generate factor returns
    factor_returns = np.zeros((n_samples, n_factors))
    for j in range(n_factors):
        factor_returns[:, j] = np.random.randn(n_samples) * factor_volatilities[j]

    # Generate idiosyncratic returns
    idio_returns = np.random.randn(n_samples, n_assets) * idio_volatility

    # Total returns: R = α + β·F + ε
    asset_returns = alphas + factor_returns @ betas.T + idio_returns

    return {
        'asset_returns': torch.tensor(asset_returns, dtype=torch.float32, device=device),
        'factor_returns': torch.tensor(factor_returns, dtype=torch.float32, device=device),
        'betas': torch.tensor(betas, dtype=torch.float32, device=device),
        'alphas': torch.tensor(alphas, dtype=torch.float32, device=device),
    }
