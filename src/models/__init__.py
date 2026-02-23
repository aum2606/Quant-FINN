"""
Neural network models for Quant-FINN.

This module contains:
- PINNBase: Base physics-informed neural network architecture
- QuantFINN: Full quantitative finance PINN with all components
- Encoders: Feature encoding networks (Fourier, temporal, sentiment)
- Layers: Physics-informed layers (GARCH, factor model, portfolio)
"""

from src.models.pinn_base import PINNBase, BlackScholesPINN
from src.models.quant_finn import QuantFINN
from src.models.encoders import (
    FourierFeatureEncoder,
    TemporalEncoder,
    FeatureFusion,
)

__all__ = [
    "PINNBase",
    "BlackScholesPINN",
    "QuantFINN",
    "FourierFeatureEncoder",
    "TemporalEncoder",
    "FeatureFusion",
]
