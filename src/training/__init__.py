"""
Training Infrastructure for Quant-FINN.

This module provides:
- PINNTrainer: Main training loop with multi-component loss
- Collocation sampling strategies
- Curriculum learning schedules
"""

from src.training.trainer import PINNTrainer
from src.training.collocation import (
    UniformSampler,
    LatinHypercubeSampler,
    AdaptiveSampler,
    ImportanceSampler,
)

__all__ = [
    "PINNTrainer",
    "UniformSampler",
    "LatinHypercubeSampler",
    "AdaptiveSampler",
    "ImportanceSampler",
]
