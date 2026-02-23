"""
Loss Functions for PINN Training.

This module provides:
- PDE residual losses (Black-Scholes, Heston, etc.)
- Economic constraint losses (no-arbitrage, monotonicity)
- Adaptive loss weighting strategies
"""

from src.losses.pde_residuals import (
    BlackScholesResidual,
    HestonResidual,
    compute_black_scholes_residual,
)
from src.losses.economic_constraints import (
    NoArbitrageLoss,
    MonotonicityLoss,
    PutCallParityLoss,
)
from src.losses.loss_weighting import (
    FixedWeighting,
    AdaptiveWeighting,
    GradientNormWeighting,
    AnnealingScheduler,
)

__all__ = [
    "BlackScholesResidual",
    "HestonResidual",
    "compute_black_scholes_residual",
    "NoArbitrageLoss",
    "MonotonicityLoss",
    "PutCallParityLoss",
    "FixedWeighting",
    "AdaptiveWeighting",
    "GradientNormWeighting",
    "AnnealingScheduler",
]
