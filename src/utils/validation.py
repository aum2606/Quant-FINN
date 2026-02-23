"""
Validation Utilities for PINN Models.

This module provides functions to validate that PINN models:
1. Satisfy the governing PDEs (low residuals)
2. Match known analytical solutions (Black-Scholes)
3. Produce consistent Greeks (autodiff vs finite difference)
4. Satisfy boundary and terminal conditions

These validations are crucial because PINNs can appear to train well
(low loss) while still violating physical constraints.
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
import numpy as np

from src.utils.autodiff import compute_gradient, compute_second_derivative


def validate_greeks_consistency(
    network: nn.Module,
    S: Tensor,
    t: Tensor,
    dS: float = 0.01,
    dt: float = 0.001,
    relative_tolerance: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Validate that network Greeks match finite difference approximations.

    Greeks (Delta, Gamma, Theta) computed via autodiff should closely match
    those computed via finite differences. Large discrepancies indicate
    potential issues with the network or training.

    Args:
        network: Trained PINN model.
        S: Stock prices tensor (batch,).
        t: Time values tensor (batch,).
        dS: Step size for finite difference in S direction.
        dt: Step size for finite difference in t direction.
        relative_tolerance: Maximum acceptable relative error.

    Returns:
        Dict containing:
            - 'delta': {'autodiff': value, 'finite_diff': value, 'error': value}
            - 'gamma': {'autodiff': value, 'finite_diff': value, 'error': value}
            - 'theta': {'autodiff': value, 'finite_diff': value, 'error': value}
            - 'passed': bool indicating if all errors are within tolerance
    """
    network.eval()
    results = {}

    # Ensure gradients are enabled
    S = S.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        # Forward pass
        V = network(S, t)

        # === Delta (∂V/∂S) ===
        delta_ad = compute_gradient(V, S)

        # Finite difference delta: (V(S+dS) - V(S-dS)) / (2*dS)
        with torch.no_grad():
            V_up = network(S + dS, t)
            V_down = network(S - dS, t)
            delta_fd = (V_up - V_down) / (2 * dS)

        delta_error = (delta_ad - delta_fd).abs().mean() / (delta_fd.abs().mean() + 1e-8)
        results['delta'] = {
            'autodiff': delta_ad.mean().item(),
            'finite_diff': delta_fd.mean().item(),
            'relative_error': delta_error.item(),
        }

        # === Gamma (∂²V/∂S²) ===
        gamma_ad = compute_second_derivative(V, S)

        # Finite difference gamma: (V(S+dS) - 2V + V(S-dS)) / dS²
        with torch.no_grad():
            V_center = network(S, t)
            gamma_fd = (V_up - 2 * V_center + V_down) / (dS ** 2)

        gamma_error = (gamma_ad - gamma_fd).abs().mean() / (gamma_fd.abs().mean() + 1e-8)
        results['gamma'] = {
            'autodiff': gamma_ad.mean().item(),
            'finite_diff': gamma_fd.mean().item(),
            'relative_error': gamma_error.item(),
        }

        # === Theta (∂V/∂t) ===
        theta_ad = compute_gradient(V, t)

        # Finite difference theta: (V(t+dt) - V(t)) / dt
        with torch.no_grad():
            V_future = network(S, t + dt)
            theta_fd = (V_future - V_center) / dt

        theta_error = (theta_ad - theta_fd).abs().mean() / (theta_fd.abs().mean() + 1e-8)
        results['theta'] = {
            'autodiff': theta_ad.mean().item(),
            'finite_diff': theta_fd.mean().item(),
            'relative_error': theta_error.item(),
        }

    # Check if all within tolerance
    results['passed'] = all(
        results[greek]['relative_error'] < relative_tolerance
        for greek in ['delta', 'gamma', 'theta']
    )

    return results


def validate_pde_residual(
    network: nn.Module,
    pde_residual_fn: Callable,
    collocation_points: Dict[str, Tensor],
    threshold: float = 1e-3,
) -> Dict[str, float]:
    """
    Validate that PDE residual is small on test collocation points.

    The PDE residual should be close to zero if the network truly
    satisfies the governing equation. This tests on points not seen
    during training.

    Args:
        network: Trained PINN model.
        pde_residual_fn: Function that computes PDE residual.
        collocation_points: Dict of input tensors at test points.
        threshold: Maximum acceptable mean absolute residual.

    Returns:
        Dict with residual statistics and pass/fail status.
    """
    network.eval()

    with torch.enable_grad():
        residual = pde_residual_fn(network, **collocation_points)

    residual_abs = residual.abs()

    results = {
        'mean_residual': residual_abs.mean().item(),
        'max_residual': residual_abs.max().item(),
        'std_residual': residual_abs.std().item(),
        'median_residual': residual_abs.median().item(),
        'threshold': threshold,
        'passed': residual_abs.mean().item() < threshold,
    }

    return results


def validate_boundary_conditions(
    network: nn.Module,
    boundary_points: Dict[str, Tensor],
    expected_values: Tensor,
    tolerance: float = 0.01,
) -> Dict[str, float]:
    """
    Validate that boundary conditions are satisfied.

    For option pricing, typical boundary conditions include:
    - V(0, t) = 0 for calls (stock worth nothing)
    - V(S, T) = max(S-K, 0) for calls at expiration

    Args:
        network: Trained PINN model.
        boundary_points: Dict of input tensors at boundary.
        expected_values: Expected output values at boundary.
        tolerance: Maximum acceptable relative error.

    Returns:
        Dict with boundary error statistics.
    """
    network.eval()

    with torch.no_grad():
        predictions = network(**boundary_points)
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)

    error = (predictions - expected_values).abs()
    relative_error = error / (expected_values.abs() + 1e-8)

    results = {
        'mean_absolute_error': error.mean().item(),
        'max_absolute_error': error.max().item(),
        'mean_relative_error': relative_error.mean().item(),
        'max_relative_error': relative_error.max().item(),
        'passed': relative_error.mean().item() < tolerance,
    }

    return results


def validate_terminal_condition(
    network: nn.Module,
    S: Tensor,
    T: float,
    K: float,
    option_type: str = 'call',
    tolerance: float = 0.01,
) -> Dict[str, float]:
    """
    Validate terminal condition for European options.

    At expiration (t=T), option value should equal payoff:
    - Call: max(S - K, 0)
    - Put: max(K - S, 0)

    Args:
        network: Trained PINN model.
        S: Stock prices at which to test.
        T: Time to maturity (terminal time).
        K: Strike price.
        option_type: 'call' or 'put'.
        tolerance: Maximum acceptable relative error.

    Returns:
        Dict with terminal condition validation results.
    """
    network.eval()
    t = torch.full_like(S, T)

    # Compute expected payoff
    if option_type == 'call':
        expected = torch.relu(S - K)
    else:
        expected = torch.relu(K - S)

    with torch.no_grad():
        predicted = network(S, t)
        if predicted.dim() > 1:
            predicted = predicted.squeeze(-1)

    error = (predicted - expected).abs()
    # Avoid division by zero for OTM options
    relative_error = error / (expected.abs() + K * 0.01)

    results = {
        'mean_absolute_error': error.mean().item(),
        'max_absolute_error': error.max().item(),
        'mean_relative_error': relative_error.mean().item(),
        'passed': relative_error.mean().item() < tolerance,
    }

    return results


def validate_monotonicity(
    network: nn.Module,
    S: Tensor,
    t: Tensor,
    option_type: str = 'call',
) -> Dict[str, bool]:
    """
    Validate monotonicity constraints for options.

    Economic constraints that must hold:
    - Call delta > 0 (call value increases with stock price)
    - Put delta < 0 (put value decreases with stock price)
    - Gamma > 0 (convexity in stock price)
    - Vega > 0 (value increases with volatility)

    Args:
        network: Trained PINN model.
        S: Stock prices tensor.
        t: Time values tensor.
        option_type: 'call' or 'put'.

    Returns:
        Dict indicating which constraints are satisfied.
    """
    network.eval()
    S = S.clone().requires_grad_(True)

    with torch.enable_grad():
        V = network(S, t)
        delta = compute_gradient(V, S)
        gamma = compute_second_derivative(V, S)

    if option_type == 'call':
        delta_valid = (delta > -0.01).all().item()  # Allow small numerical errors
    else:
        delta_valid = (delta < 0.01).all().item()

    gamma_valid = (gamma > -0.01).all().item()

    results = {
        'delta_monotonicity': delta_valid,
        'gamma_convexity': gamma_valid,
        'all_passed': delta_valid and gamma_valid,
    }

    return results


def validate_put_call_parity(
    call_network: nn.Module,
    put_network: nn.Module,
    S: Tensor,
    t: Tensor,
    K: float,
    r: float,
    T: float,
    tolerance: float = 0.01,
) -> Dict[str, float]:
    """
    Validate put-call parity: C - P = S - K*exp(-r*(T-t)).

    This is a fundamental no-arbitrage condition. Violation indicates
    inconsistent pricing between calls and puts.

    Args:
        call_network: Network pricing calls.
        put_network: Network pricing puts.
        S: Stock prices.
        t: Current times.
        K: Strike price.
        r: Risk-free rate.
        T: Expiration time.
        tolerance: Maximum acceptable violation as fraction of stock price.

    Returns:
        Dict with parity violation statistics.
    """
    call_network.eval()
    put_network.eval()

    with torch.no_grad():
        C = call_network(S, t).squeeze()
        P = put_network(S, t).squeeze()

    # Put-call parity
    lhs = C - P
    rhs = S - K * torch.exp(-r * (T - t))
    violation = (lhs - rhs).abs()
    relative_violation = violation / S

    results = {
        'mean_violation': violation.mean().item(),
        'max_violation': violation.max().item(),
        'mean_relative_violation': relative_violation.mean().item(),
        'passed': relative_violation.mean().item() < tolerance,
    }

    return results


def compare_to_black_scholes(
    network: nn.Module,
    S: Tensor,
    t: Tensor,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = 'call',
    tolerance: float = 0.05,
) -> Dict[str, float]:
    """
    Compare network output to analytical Black-Scholes prices.

    This is a sanity check when the network is trained on Black-Scholes
    dynamics - it should closely match the analytical solution.

    Args:
        network: Trained PINN model.
        S: Stock prices.
        t: Current times (time to expiration = T - t).
        K: Strike price.
        r: Risk-free rate.
        sigma: Volatility.
        T: Expiration time.
        option_type: 'call' or 'put'.
        tolerance: Maximum acceptable relative error.

    Returns:
        Dict with comparison statistics.
    """
    from scipy.stats import norm

    # Compute analytical Black-Scholes price
    tau = T - t.detach().cpu().numpy()  # Time to expiration
    S_np = S.detach().cpu().numpy()

    # Avoid division by zero for tau = 0
    tau = np.maximum(tau, 1e-8)

    d1 = (np.log(S_np / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if option_type == 'call':
        bs_price = S_np * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        bs_price = K * np.exp(-r * tau) * norm.cdf(-d2) - S_np * norm.cdf(-d1)

    bs_price = torch.tensor(bs_price, dtype=S.dtype, device=S.device)

    # Network prediction
    network.eval()
    with torch.no_grad():
        network_price = network(S, t).squeeze()

    error = (network_price - bs_price).abs()
    relative_error = error / (bs_price.abs() + 1e-8)

    results = {
        'mean_absolute_error': error.mean().item(),
        'max_absolute_error': error.max().item(),
        'mean_relative_error': relative_error.mean().item(),
        'max_relative_error': relative_error.max().item(),
        'passed': relative_error.mean().item() < tolerance,
    }

    return results


def full_pinn_validation(
    network: nn.Module,
    pde_residual_fn: Callable,
    test_data: Dict[str, Tensor],
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> Dict[str, Dict]:
    """
    Run comprehensive validation suite for a trained PINN.

    This combines multiple validation checks into a single report.

    Args:
        network: Trained PINN model.
        pde_residual_fn: Function computing PDE residual.
        test_data: Dict containing 'S', 't' test tensors.
        K: Strike price.
        r: Risk-free rate.
        sigma: Volatility.
        T: Expiration time.

    Returns:
        Comprehensive validation report.
    """
    S = test_data['S']
    t = test_data['t']

    results = {}

    # 1. Greeks consistency
    results['greeks'] = validate_greeks_consistency(network, S, t)

    # 2. PDE residual
    results['pde_residual'] = validate_pde_residual(
        network, pde_residual_fn,
        {'S': S, 't': t, 'sigma': sigma, 'r': r}
    )

    # 3. Terminal condition
    results['terminal'] = validate_terminal_condition(network, S, T, K)

    # 4. Monotonicity
    results['monotonicity'] = validate_monotonicity(network, S, t)

    # 5. Black-Scholes comparison
    results['black_scholes'] = compare_to_black_scholes(
        network, S, t, K, r, sigma, T
    )

    # Overall pass/fail
    results['all_passed'] = all([
        results['greeks']['passed'],
        results['pde_residual']['passed'],
        results['terminal']['passed'],
        results['monotonicity']['all_passed'],
        results['black_scholes']['passed'],
    ])

    return results
