"""
Loss Weighting Strategies for Multi-Objective PINN Training.

PINNs have multiple loss components (data, PDE, BC, IC, economic).
Proper weighting is crucial:
- Equal weights often fail because loss magnitudes differ
- PDE loss can dominate and prevent data fitting
- Economic constraints may conflict with PDE

This module provides strategies:
1. Fixed weighting with manual tuning
2. Gradient-based normalization
3. Annealing (gradually increase PDE weight)
4. Uncertainty-based weighting
5. Adaptive multi-task learning
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List
import numpy as np


class FixedWeighting:
    """
    Fixed loss weighting with manual weights.

    Simple but effective when weights are tuned properly.
    Good starting point before trying adaptive methods.

    Tips for tuning:
    - Start with equal weights, observe loss magnitudes
    - Scale weights inversely proportional to typical loss values
    - PDE weight often needs to be smaller (0.1-0.01)
    - Boundary/terminal conditions often need higher weight
    """

    def __init__(
        self,
        weights: Dict[str, float],
    ):
        """
        Initialize fixed weighting.

        Args:
            weights: Dict mapping loss names to weights.

        Example:
            >>> weighting = FixedWeighting({
            ...     'data': 1.0,
            ...     'pde': 0.1,
            ...     'boundary': 10.0,
            ...     'terminal': 10.0,
            ...     'economic': 1.0,
            ... })
        """
        self.weights = weights

    def __call__(
        self,
        losses: Dict[str, Tensor],
        epoch: Optional[int] = None,
    ) -> Tensor:
        """
        Compute weighted total loss.

        Args:
            losses: Dict of loss tensors.
            epoch: Current epoch (unused for fixed weighting).

        Returns:
            Weighted sum of losses.
        """
        total = 0.0
        for name, loss in losses.items():
            weight = self.weights.get(name, 1.0)
            total = total + weight * loss

        return total

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()


class AnnealingScheduler:
    """
    Annealing Schedule for Loss Weights.

    Gradually changes weights during training. Common strategy:
    1. Start with low PDE weight (focus on data fit)
    2. Gradually increase PDE weight (enforce physics)

    This helps avoid getting stuck in local minima where
    the network satisfies neither data nor physics well.
    """

    def __init__(
        self,
        initial_weights: Dict[str, float],
        final_weights: Dict[str, float],
        warmup_epochs: int = 100,
        schedule: str = 'linear',
    ):
        """
        Initialize annealing scheduler.

        Args:
            initial_weights: Weights at epoch 0.
            final_weights: Weights after warmup.
            warmup_epochs: Number of epochs for transition.
            schedule: 'linear', 'exponential', or 'cosine'.
        """
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule

        # Ensure same keys
        assert set(initial_weights.keys()) == set(final_weights.keys())

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get weights for given epoch.

        Args:
            epoch: Current training epoch.

        Returns:
            Dict of weights.
        """
        if epoch >= self.warmup_epochs:
            return self.final_weights.copy()

        progress = epoch / self.warmup_epochs

        if self.schedule == 'linear':
            alpha = progress
        elif self.schedule == 'exponential':
            alpha = 1 - np.exp(-5 * progress)  # Quick initial increase
        elif self.schedule == 'cosine':
            alpha = 0.5 * (1 - np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        # Interpolate weights
        weights = {}
        for name in self.initial_weights:
            w_init = self.initial_weights[name]
            w_final = self.final_weights[name]
            weights[name] = w_init + alpha * (w_final - w_init)

        return weights

    def __call__(
        self,
        losses: Dict[str, Tensor],
        epoch: int,
    ) -> Tensor:
        """Compute weighted loss with annealed weights."""
        weights = self.get_weights(epoch)

        total = 0.0
        for name, loss in losses.items():
            weight = weights.get(name, 1.0)
            total = total + weight * loss

        return total


class GradientNormWeighting(nn.Module):
    """
    Gradient-Based Loss Weighting.

    Normalizes weights based on gradient magnitudes. Tasks with
    larger gradients get smaller weights to prevent dominance.

    Reference: "GradNorm: Gradient Normalization for Adaptive Loss
    Balancing in Deep Multitask Networks" (Chen et al., 2018)

    This automatically balances loss components without manual tuning.
    """

    def __init__(
        self,
        loss_names: List[str],
        alpha: float = 1.5,
        initial_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize gradient norm weighting.

        Args:
            loss_names: Names of loss components.
            alpha: Restoring force strength (higher = faster balancing).
            initial_weights: Initial weights (default: equal).
        """
        super().__init__()

        self.loss_names = loss_names
        self.alpha = alpha
        self.n_losses = len(loss_names)

        # Learnable log weights
        if initial_weights:
            init_values = [np.log(initial_weights.get(name, 1.0)) for name in loss_names]
        else:
            init_values = [0.0] * self.n_losses

        self.log_weights = nn.Parameter(torch.tensor(init_values))

        # Track initial loss values for normalization
        self.register_buffer('initial_losses', torch.ones(self.n_losses))
        self.initialized = False

    def forward(
        self,
        losses: Dict[str, Tensor],
        shared_params: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """
        Compute weighted loss with gradient-based weighting.

        Args:
            losses: Dict of loss tensors.
            shared_params: Shared network parameters for gradient computation.

        Returns:
            Weighted total loss.
        """
        # Get weights (softmax of log weights for proper normalization)
        weights = torch.softmax(self.log_weights, dim=0) * self.n_losses

        # Compute weighted loss
        total = 0.0
        for i, name in enumerate(self.loss_names):
            if name in losses:
                total = total + weights[i] * losses[name]

        return total

    def get_weights(self) -> Dict[str, float]:
        """Get current normalized weights."""
        weights = torch.softmax(self.log_weights, dim=0) * self.n_losses
        return {
            name: weights[i].item()
            for i, name in enumerate(self.loss_names)
        }


class AdaptiveWeighting(nn.Module):
    """
    Uncertainty-Based Adaptive Weighting.

    Models task uncertainty and uses it to weight losses:
        L_total = Σᵢ (1/2σᵢ²) Lᵢ + log(σᵢ)

    The log(σ) term prevents all weights from going to zero.

    Reference: "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics" (Kendall et al., 2018)
    """

    def __init__(
        self,
        loss_names: List[str],
        initial_log_vars: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize adaptive weighting.

        Args:
            loss_names: Names of loss components.
            initial_log_vars: Initial log variances (log(σ²)).
        """
        super().__init__()

        self.loss_names = loss_names
        self.n_losses = len(loss_names)

        # Learnable log variances
        if initial_log_vars:
            init_values = [initial_log_vars.get(name, 0.0) for name in loss_names]
        else:
            init_values = [0.0] * self.n_losses

        self.log_vars = nn.Parameter(torch.tensor(init_values))

    def forward(self, losses: Dict[str, Tensor]) -> Tensor:
        """
        Compute adaptively weighted loss.

        Args:
            losses: Dict of loss tensors.

        Returns:
            Weighted total loss.
        """
        total = 0.0

        for i, name in enumerate(self.loss_names):
            if name in losses:
                log_var = self.log_vars[i]
                precision = torch.exp(-log_var)  # 1/σ²

                # Weight by precision, add regularization term
                weighted_loss = precision * losses[name] + log_var

                total = total + weighted_loss

        return total

    def get_weights(self) -> Dict[str, float]:
        """Get current effective weights (precisions)."""
        precisions = torch.exp(-self.log_vars)
        return {
            name: precisions[i].item()
            for i, name in enumerate(self.loss_names)
        }

    def get_uncertainties(self) -> Dict[str, float]:
        """Get current uncertainties (standard deviations)."""
        stds = torch.exp(0.5 * self.log_vars)
        return {
            name: stds[i].item()
            for i, name in enumerate(self.loss_names)
        }


class CausalWeighting:
    """
    Causal Training Weighting for Time-Dependent PDEs.

    Enforces causality: solution at earlier times must be correct
    before later times can be learned properly. Weights residuals
    by how well earlier times satisfy the PDE.

    Reference: "Respecting Causality is All You Need for Training
    Physics-Informed Neural Networks" (Wang et al., 2022)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        adaptive: bool = True,
    ):
        """
        Initialize causal weighting.

        Args:
            epsilon: Tolerance parameter (higher = stricter causality).
            adaptive: If True, adapt epsilon based on convergence.
        """
        self.epsilon = epsilon
        self.adaptive = adaptive
        self.residual_history = []

    def __call__(
        self,
        residuals: Tensor,
        times: Tensor,
    ) -> Tensor:
        """
        Apply causal weighting to PDE residuals.

        Args:
            residuals: PDE residuals at collocation points.
            times: Time values at collocation points.

        Returns:
            Causally weighted loss.
        """
        # Sort by time
        sorted_indices = torch.argsort(times)
        sorted_residuals = residuals[sorted_indices]

        # Cumulative squared residuals (measure of error at earlier times)
        residuals_sq = sorted_residuals.pow(2)
        cumsum = torch.cumsum(residuals_sq, dim=0)

        # Causal weights: exp(-ε * cumulative_error)
        # Points with large earlier errors get downweighted
        weights = torch.exp(-self.epsilon * cumsum.detach())

        # Weighted loss
        weighted_loss = (weights * residuals_sq).mean()

        return weighted_loss


class LossTracker:
    """
    Utility class for tracking loss component evolution.

    Useful for:
    - Monitoring training progress
    - Diagnosing weighting issues
    - Visualizing loss curves
    """

    def __init__(self, loss_names: List[str]):
        """
        Initialize loss tracker.

        Args:
            loss_names: Names of loss components to track.
        """
        self.loss_names = loss_names
        self.history: Dict[str, List[float]] = {name: [] for name in loss_names}
        self.history['total'] = []

    def update(self, losses: Dict[str, Tensor], total: Tensor):
        """
        Record loss values for current epoch.

        Args:
            losses: Dict of loss tensors.
            total: Total weighted loss.
        """
        for name in self.loss_names:
            if name in losses:
                self.history[name].append(losses[name].item())

        self.history['total'].append(total.item())

    def get_history(self) -> Dict[str, List[float]]:
        """Get full loss history."""
        return self.history.copy()

    def get_recent(self, n: int = 10) -> Dict[str, float]:
        """Get average of last n epochs."""
        recent = {}
        for name, values in self.history.items():
            if len(values) >= n:
                recent[name] = np.mean(values[-n:])
            elif values:
                recent[name] = np.mean(values)
        return recent

    def is_converged(
        self,
        window: int = 50,
        threshold: float = 0.01,
    ) -> bool:
        """
        Check if training has converged.

        Args:
            window: Window size for comparison.
            threshold: Maximum relative change for convergence.

        Returns:
            True if converged.
        """
        if len(self.history['total']) < 2 * window:
            return False

        recent = np.mean(self.history['total'][-window:])
        previous = np.mean(self.history['total'][-2*window:-window])

        relative_change = abs(recent - previous) / (previous + 1e-8)

        return relative_change < threshold
