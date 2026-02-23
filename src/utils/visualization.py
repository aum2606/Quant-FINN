"""
Visualization Utilities for PINN Models.

This module provides diagnostic plotting functions for:
1. Training loss curves (multi-component)
2. Solution surfaces V(S, t)
3. Greeks (delta, gamma, theta)
4. Terminal condition comparison

All plots use matplotlib with a clean financial style.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Tuple, Optional, List


def plot_training_curves(history: Dict[str, List[float]]) -> plt.Figure:
    """
    Plot training loss curves for each component.

    Args:
        history: Dict mapping loss name to list of values per epoch.
                 Expected keys: 'total_loss', 'pde_loss', 'boundary_loss',
                 'terminal_loss', etc.

    Returns:
        Matplotlib Figure with loss curves.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2']

    for idx, (name, values) in enumerate(history.items()):
        color = colors[idx % len(colors)]
        ax.plot(values, label=name, color=color, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Components', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_solution_surface(
    network,
    S_range: Tuple[float, float],
    t_range: Tuple[float, float],
    n_points: int = 50,
) -> plt.Figure:
    """
    Plot 3D surface of option value V(S, t).

    Args:
        network: Trained PINN network with forward(S, t).
        S_range: (min, max) stock price range.
        t_range: (min, max) time range.
        n_points: Grid resolution.

    Returns:
        Matplotlib Figure with 3D surface plot.
    """
    S_vals = np.linspace(S_range[0], S_range[1], n_points)
    t_vals = np.linspace(t_range[0], t_range[1], n_points)
    S_grid, t_grid = np.meshgrid(S_vals, t_vals)

    # Flatten for network evaluation
    S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32)
    t_flat = torch.tensor(t_grid.flatten(), dtype=torch.float32)

    # Move to same device as network
    device = next(network.parameters()).device
    S_flat = S_flat.to(device)
    t_flat = t_flat.to(device)

    # Evaluate network
    network.eval()
    with torch.no_grad():
        V_flat = network(S_flat, t_flat).squeeze().cpu().numpy()

    V_grid = V_flat.reshape(S_grid.shape)

    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        S_grid, t_grid, V_grid,
        cmap=cm.viridis,
        alpha=0.8,
        edgecolor='none',
    )

    ax.set_xlabel('Stock Price (S)', fontsize=11)
    ax.set_ylabel('Time (t)', fontsize=11)
    ax.set_zlabel('Option Value (V)', fontsize=11)
    ax.set_title('PINN Option Price Surface V(S, t)', fontsize=14)

    fig.colorbar(surf, shrink=0.5, aspect=10, label='V(S,t)')
    plt.tight_layout()
    return fig


def plot_greeks(
    network,
    S_range: Tuple[float, float],
    t_fixed: float = 0.5,
    n_points: int = 200,
) -> plt.Figure:
    """
    Plot Greeks (delta, gamma, theta) as functions of stock price.

    Args:
        network: Trained PINN with compute_greeks(S, t) method.
        S_range: (min, max) stock price range.
        t_fixed: Fixed time at which to evaluate Greeks.
        n_points: Number of evaluation points.

    Returns:
        Matplotlib Figure with Greeks plots.
    """
    device = next(network.parameters()).device

    S = torch.linspace(S_range[0], S_range[1], n_points, device=device)
    t = torch.full_like(S, t_fixed)

    # Enable gradients for Greek computation
    S_grad = S.clone().requires_grad_(True)
    t_grad = t.clone().requires_grad_(True)

    network.eval()
    delta, gamma, theta, V = network.compute_greeks(S_grad, t_grad)

    # Move to CPU for plotting
    S_np = S.cpu().numpy()
    delta_np = delta.detach().cpu().numpy()
    gamma_np = gamma.detach().cpu().numpy()
    theta_np = theta.detach().cpu().numpy()
    V_np = V.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Option Value
    axes[0, 0].plot(S_np, V_np, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Stock Price (S)')
    axes[0, 0].set_ylabel('V(S, t)')
    axes[0, 0].set_title(f'Option Value (t={t_fixed})')
    axes[0, 0].grid(True, alpha=0.3)

    # Delta
    axes[0, 1].plot(S_np, delta_np, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Stock Price (S)')
    axes[0, 1].set_ylabel('Delta')
    axes[0, 1].set_title('Delta (dV/dS)')
    axes[0, 1].grid(True, alpha=0.3)

    # Gamma
    axes[1, 0].plot(S_np, gamma_np, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Stock Price (S)')
    axes[1, 0].set_ylabel('Gamma')
    axes[1, 0].set_title('Gamma (d2V/dS2)')
    axes[1, 0].grid(True, alpha=0.3)

    # Theta
    axes[1, 1].plot(S_np, theta_np, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Stock Price (S)')
    axes[1, 1].set_ylabel('Theta')
    axes[1, 1].set_title('Theta (dV/dt)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Option Greeks at t = {t_fixed}', fontsize=14)
    plt.tight_layout()
    return fig


def plot_terminal_condition(
    network,
    S_range: Tuple[float, float],
    T: float,
    K: float,
    option_type: str = 'call',
    n_points: int = 200,
) -> plt.Figure:
    """
    Plot network output at expiry vs analytical payoff.

    Args:
        network: Trained PINN network.
        S_range: (min, max) stock price range.
        T: Time to expiry.
        K: Strike price.
        option_type: 'call' or 'put'.
        n_points: Number of evaluation points.

    Returns:
        Matplotlib Figure comparing payoff to network output.
    """
    device = next(network.parameters()).device

    S = torch.linspace(S_range[0], S_range[1], n_points, device=device)
    t = torch.full_like(S, T)  # At expiry

    # Compute payoff
    if option_type == 'call':
        payoff = torch.clamp(S - K, min=0).cpu().numpy()
    else:
        payoff = torch.clamp(K - S, min=0).cpu().numpy()

    # Network prediction at expiry
    network.eval()
    with torch.no_grad():
        V_pred = network(S, t).squeeze().cpu().numpy()

    S_np = S.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(S_np, payoff, 'b-', linewidth=2, label='Analytical Payoff')
    ax.plot(S_np, V_pred, 'r--', linewidth=2, label='PINN Prediction')

    ax.axvline(x=K, color='gray', linestyle=':', alpha=0.5, label=f'Strike K={K}')

    ax.set_xlabel('Stock Price (S)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(
        f'Terminal Condition: {option_type.capitalize()} Option at T={T}',
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Compute and display error
    error = np.abs(V_pred - payoff)
    ax.fill_between(S_np, payoff, V_pred, alpha=0.2, color='red', label='Error')

    mae = np.mean(error)
    ax.text(
        0.05, 0.95,
        f'MAE: {mae:.4f}',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    plt.tight_layout()
    return fig
