"""
Collocation Point Sampling Strategies.

Collocation points are locations in the domain where we enforce
the PDE constraint (residual = 0). Proper sampling is critical:

1. Coverage: Points should cover the entire domain
2. Concentration: More points where solution has high curvature
3. Adaptivity: Add points where residual is large

Sampling strategies in order of sophistication:
- Uniform: Simple random sampling
- Latin Hypercube: Better space coverage
- Importance: More points near critical regions
- Adaptive: Dynamic allocation based on residuals
"""

import torch
from torch import Tensor
from typing import Dict, Tuple, Optional, Callable
import numpy as np


class UniformSampler:
    """
    Uniform Random Sampling.

    Samples points uniformly at random within bounds.
    Simple and fast, but may miss important regions.

    Good for:
    - Initial exploration
    - Simple problems
    - Baseline comparisons
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        device: str = 'cpu',
    ):
        """
        Initialize uniform sampler.

        Args:
            bounds: Dict mapping variable names to (min, max) bounds.
            device: Device for generated tensors.

        Example:
            >>> sampler = UniformSampler({
            ...     'S': (50.0, 150.0),  # Stock price
            ...     't': (0.0, 1.0),      # Time
            ... })
        """
        self.bounds = bounds
        self.device = device
        self.dim = len(bounds)
        self.var_names = list(bounds.keys())

    def sample(self, n_points: int) -> Dict[str, Tensor]:
        """
        Sample n_points uniformly from the domain.

        Args:
            n_points: Number of points to sample.

        Returns:
            Dict mapping variable names to sampled tensors.
        """
        samples = {}

        for name in self.var_names:
            low, high = self.bounds[name]
            samples[name] = (
                torch.rand(n_points, device=self.device) * (high - low) + low
            )

        return samples

    def sample_boundary(
        self,
        n_points: int,
        boundary: str,
    ) -> Dict[str, Tensor]:
        """
        Sample points on a specific boundary.

        Args:
            n_points: Number of points.
            boundary: Variable name to fix at boundary.

        Returns:
            Dict with boundary points.
        """
        samples = self.sample(n_points)

        # Fix the boundary variable
        low, high = self.bounds[boundary]

        # Randomly choose lower or upper boundary
        boundary_values = torch.where(
            torch.rand(n_points, device=self.device) < 0.5,
            torch.full((n_points,), low, device=self.device),
            torch.full((n_points,), high, device=self.device),
        )

        samples[boundary] = boundary_values

        return samples


class LatinHypercubeSampler:
    """
    Latin Hypercube Sampling (LHS).

    LHS provides better coverage than uniform sampling by ensuring
    samples are spread evenly across each dimension.

    For each dimension, divide into n_points intervals and sample
    exactly once from each interval.

    Better for:
    - Higher-dimensional problems
    - When good coverage is important
    - Smaller sample sizes
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        device: str = 'cpu',
        seed: Optional[int] = None,
    ):
        """
        Initialize LHS sampler.

        Args:
            bounds: Dict mapping variable names to (min, max) bounds.
            device: Device for generated tensors.
            seed: Random seed for reproducibility.
        """
        self.bounds = bounds
        self.device = device
        self.dim = len(bounds)
        self.var_names = list(bounds.keys())
        self.rng = np.random.default_rng(seed)

    def sample(self, n_points: int) -> Dict[str, Tensor]:
        """
        Sample n_points using Latin Hypercube Sampling.

        Args:
            n_points: Number of points to sample.

        Returns:
            Dict mapping variable names to sampled tensors.
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
            unit_samples = sampler.random(n=n_points)
        except ImportError:
            # Fallback to manual LHS
            unit_samples = self._manual_lhs(n_points)

        samples = {}
        for i, name in enumerate(self.var_names):
            low, high = self.bounds[name]
            scaled = unit_samples[:, i] * (high - low) + low
            samples[name] = torch.tensor(scaled, dtype=torch.float32, device=self.device)

        return samples

    def _manual_lhs(self, n_points: int) -> np.ndarray:
        """Manual LHS implementation without scipy."""
        samples = np.zeros((n_points, self.dim))

        for d in range(self.dim):
            # Create intervals
            intervals = np.arange(n_points) / n_points

            # Add uniform noise within each interval
            samples[:, d] = intervals + self.rng.uniform(0, 1/n_points, n_points)

            # Shuffle to break correlation
            self.rng.shuffle(samples[:, d])

        return samples


class ImportanceSampler:
    """
    Importance Sampling for Collocation Points.

    Samples more points in regions where the solution has
    high curvature or importance:
    - Near the strike price (for options)
    - Near boundaries
    - Near time t=0 or t=T

    Uses a proposal distribution that concentrates mass
    in important regions.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        concentration_points: Optional[Dict[str, float]] = None,
        concentration_scale: float = 0.3,
        device: str = 'cpu',
    ):
        """
        Initialize importance sampler.

        Args:
            bounds: Variable bounds.
            concentration_points: Points to concentrate around.
            concentration_scale: Width of concentration (fraction of range).
            device: Device for tensors.

        Example:
            >>> # Concentrate around strike K=100
            >>> sampler = ImportanceSampler(
            ...     bounds={'S': (50, 150), 't': (0, 1)},
            ...     concentration_points={'S': 100.0},
            ... )
        """
        self.bounds = bounds
        self.concentration_points = concentration_points or {}
        self.concentration_scale = concentration_scale
        self.device = device
        self.var_names = list(bounds.keys())

    def sample(self, n_points: int) -> Dict[str, Tensor]:
        """
        Sample using importance sampling.

        Uses a mixture of uniform and concentrated distributions.
        """
        samples = {}

        for name in self.var_names:
            low, high = self.bounds[name]
            range_size = high - low

            if name in self.concentration_points:
                # Mixture of uniform and Gaussian around concentration point
                center = self.concentration_points[name]
                scale = range_size * self.concentration_scale

                # 50% uniform, 50% concentrated
                n_uniform = n_points // 2
                n_concentrated = n_points - n_uniform

                uniform_samples = torch.rand(n_uniform, device=self.device) * range_size + low
                concentrated_samples = torch.randn(n_concentrated, device=self.device) * scale + center

                # Clip to bounds
                concentrated_samples = torch.clamp(concentrated_samples, low, high)

                combined = torch.cat([uniform_samples, concentrated_samples])

                # Shuffle
                perm = torch.randperm(n_points, device=self.device)
                samples[name] = combined[perm]
            else:
                samples[name] = torch.rand(n_points, device=self.device) * range_size + low

        return samples


class AdaptiveSampler:
    """
    Adaptive Collocation Point Sampling.

    Dynamically adds points where PDE residual is large.
    This focuses computational effort where it's needed most.

    Algorithm:
    1. Start with uniform/LHS samples
    2. Compute residuals at all points
    3. Sample new points with probability proportional to |residual|
    4. Add noise to prevent clustering
    5. Repeat during training

    This significantly improves convergence for problems with
    localized features (e.g., near option strike).
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        initial_points: int = 1000,
        device: str = 'cpu',
    ):
        """
        Initialize adaptive sampler.

        Args:
            bounds: Variable bounds.
            initial_points: Number of initial points.
            device: Device for tensors.
        """
        self.bounds = bounds
        self.device = device
        self.var_names = list(bounds.keys())

        # Initialize with LHS
        self.base_sampler = LatinHypercubeSampler(bounds, device)
        self.current_points = self.base_sampler.sample(initial_points)
        self.n_points = initial_points

    def update(
        self,
        residuals: Tensor,
        n_new_points: int,
        noise_scale: float = 0.05,
    ) -> Dict[str, Tensor]:
        """
        Update collocation points based on residuals.

        Args:
            residuals: PDE residuals at current points.
            n_new_points: Number of new points to add.
            noise_scale: Scale of noise to add (fraction of range).

        Returns:
            Updated collocation points.
        """
        # Importance weights from residuals
        weights = residuals.abs()
        weights = weights / (weights.sum() + 1e-8)

        # Sample indices with replacement
        indices = torch.multinomial(weights, n_new_points, replacement=True)

        new_points = {}
        for name in self.var_names:
            low, high = self.bounds[name]
            range_size = high - low

            # Get selected points and add noise
            selected = self.current_points[name][indices]
            noise = torch.randn_like(selected) * (range_size * noise_scale)
            new_values = selected + noise

            # Clip to bounds
            new_values = torch.clamp(new_values, low, high)

            # Concatenate with existing
            new_points[name] = torch.cat([self.current_points[name], new_values])

        self.current_points = new_points
        self.n_points = self.n_points + n_new_points

        return new_points

    def sample(self, n_points: Optional[int] = None) -> Dict[str, Tensor]:
        """
        Get current collocation points.

        Args:
            n_points: If provided, randomly subsample to this size.

        Returns:
            Current collocation points.
        """
        if n_points is None or n_points >= self.n_points:
            return self.current_points

        # Random subsample
        indices = torch.randperm(self.n_points, device=self.device)[:n_points]

        return {
            name: self.current_points[name][indices]
            for name in self.var_names
        }

    def reset(self, n_points: Optional[int] = None):
        """Reset to initial uniform/LHS sampling."""
        n = n_points or self.n_points
        self.current_points = self.base_sampler.sample(n)
        self.n_points = n


class BoundaryConditionSampler:
    """
    Specialized sampler for boundary and terminal conditions.

    For option pricing, boundaries include:
    - S = 0: Lower boundary
    - S = S_max: Upper boundary
    - t = T: Terminal condition
    - t = 0: Initial condition (if any)
    """

    def __init__(
        self,
        S_bounds: Tuple[float, float],
        t_bounds: Tuple[float, float],
        device: str = 'cpu',
    ):
        """
        Initialize boundary sampler.

        Args:
            S_bounds: (S_min, S_max) stock price bounds.
            t_bounds: (t_min, t_max) time bounds.
            device: Device for tensors.
        """
        self.S_min, self.S_max = S_bounds
        self.t_min, self.t_max = t_bounds
        self.device = device

    def sample_terminal(self, n_points: int) -> Dict[str, Tensor]:
        """Sample points at terminal time t = t_max."""
        S = torch.rand(n_points, device=self.device) * (self.S_max - self.S_min) + self.S_min
        t = torch.full((n_points,), self.t_max, device=self.device)
        return {'S': S, 't': t}

    def sample_lower_boundary(self, n_points: int) -> Dict[str, Tensor]:
        """Sample points at lower boundary S = S_min."""
        S = torch.full((n_points,), self.S_min, device=self.device)
        t = torch.rand(n_points, device=self.device) * (self.t_max - self.t_min) + self.t_min
        return {'S': S, 't': t}

    def sample_upper_boundary(self, n_points: int) -> Dict[str, Tensor]:
        """Sample points at upper boundary S = S_max."""
        S = torch.full((n_points,), self.S_max, device=self.device)
        t = torch.rand(n_points, device=self.device) * (self.t_max - self.t_min) + self.t_min
        return {'S': S, 't': t}

    def sample_all_boundaries(
        self,
        n_terminal: int,
        n_lower: int,
        n_upper: int,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        """Sample from all boundaries."""
        return (
            self.sample_terminal(n_terminal),
            self.sample_lower_boundary(n_lower),
            self.sample_upper_boundary(n_upper),
        )
