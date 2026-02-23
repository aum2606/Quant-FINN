"""
Quant-FINN: Physics-Informed Neural Networks for Quantitative Finance

This package implements PINNs that incorporate financial domain knowledge
(PDEs, economic constraints) directly into neural network training.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.models.quant_finn import QuantFINN
from src.models.pinn_base import PINNBase
from src.training.trainer import PINNTrainer

__all__ = [
    "QuantFINN",
    "PINNBase",
    "PINNTrainer",
]
