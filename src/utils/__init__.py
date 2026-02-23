"""
Utility modules for Quant-FINN.
"""

from src.utils.autodiff import (
    compute_gradient,
    compute_hessian_diagonal,
    compute_jacobian,
)
from src.utils.normalization import Normalizer, LogNormalizer
from src.utils.visualization import (
    plot_training_curves,
    plot_solution_surface,
    plot_greeks,
    plot_terminal_condition,
)

__all__ = [
    "compute_gradient",
    "compute_hessian_diagonal",
    "compute_jacobian",
    "Normalizer",
    "LogNormalizer",
    "plot_training_curves",
    "plot_solution_surface",
    "plot_greeks",
    "plot_terminal_condition",
]

