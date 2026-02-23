"""
Data Processing Module for Quant-FINN.

This module provides utilities for:
- Loading and preprocessing financial data
- Creating training datasets
- Generating synthetic data for testing
"""

from src.data.preprocessing import (
    compute_returns,
    compute_log_returns,
    compute_realized_volatility,
    compute_technical_indicators,
)
from src.data.synthetic import (
    generate_gbm_paths,
    generate_heston_paths,
    generate_black_scholes_data,
)

__all__ = [
    "compute_returns",
    "compute_log_returns",
    "compute_realized_volatility",
    "compute_technical_indicators",
    "generate_gbm_paths",
    "generate_heston_paths",
    "generate_black_scholes_data",
]
