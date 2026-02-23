"""
Data Preprocessing Utilities for Financial Data.

This module provides functions to transform raw price data
into features suitable for PINN training:
- Returns computation
- Volatility estimation
- Technical indicators
- Normalization
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
import numpy as np


def compute_returns(
    prices: Tensor,
    method: str = 'simple',
    dim: int = -1,
) -> Tensor:
    """
    Compute returns from price series.

    Args:
        prices: Price tensor.
        method: 'simple' for (P_t - P_{t-1}) / P_{t-1}
                'log' for log(P_t / P_{t-1})
        dim: Dimension along which to compute returns.

    Returns:
        Returns tensor (one element shorter along dim).
    """
    if method == 'simple':
        returns = prices.diff(dim=dim) / prices.narrow(dim, 0, prices.size(dim) - 1)
    elif method == 'log':
        returns = torch.log(prices).diff(dim=dim)
    else:
        raise ValueError(f"Unknown method: {method}")

    return returns


def compute_log_returns(prices: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log returns from price series.

    Log returns are preferred because:
    1. Additive over time: r_{t,t+n} = r_{t,t+1} + ... + r_{t+n-1,t+n}
    2. More symmetric distribution
    3. Bounded below (can't lose more than 100%)

    Args:
        prices: Price tensor.
        dim: Dimension for computation.

    Returns:
        Log returns tensor.
    """
    return torch.log(prices).diff(dim=dim)


def compute_realized_volatility(
    returns: Tensor,
    window: int = 20,
    annualization_factor: float = 252.0,
) -> Tensor:
    """
    Compute realized volatility using rolling window.

    Realized volatility is the standard deviation of returns
    over a lookback window, annualized.

    Args:
        returns: Returns tensor (batch, seq_len) or (seq_len,).
        window: Lookback window size.
        annualization_factor: Trading days per year (252 for daily).

    Returns:
        Realized volatility tensor.
    """
    if returns.dim() == 1:
        returns = returns.unsqueeze(0)

    batch_size, seq_len = returns.shape

    # Compute rolling variance
    volatilities = []

    for i in range(seq_len - window + 1):
        window_returns = returns[:, i:i + window]
        vol = window_returns.std(dim=1) * np.sqrt(annualization_factor)
        volatilities.append(vol)

    if volatilities:
        return torch.stack(volatilities, dim=1)
    else:
        return torch.zeros(batch_size, 0, device=returns.device)


def compute_technical_indicators(
    prices: Tensor,
    high: Optional[Tensor] = None,
    low: Optional[Tensor] = None,
    volume: Optional[Tensor] = None,
) -> dict:
    """
    Compute common technical indicators.

    Args:
        prices: Close prices (batch, seq_len) or (seq_len,).
        high: High prices (optional).
        low: Low prices (optional).
        volume: Volume (optional).

    Returns:
        Dict of indicator tensors.
    """
    if prices.dim() == 1:
        prices = prices.unsqueeze(0)

    indicators = {}

    # Simple Moving Averages
    for window in [5, 10, 20, 50]:
        if prices.size(1) >= window:
            sma = _rolling_mean(prices, window)
            indicators[f'sma_{window}'] = sma

            # Price relative to SMA (mean reversion indicator)
            indicators[f'price_sma_{window}_ratio'] = prices / (sma + 1e-8)

    # Exponential Moving Average
    for span in [12, 26]:
        if prices.size(1) >= span:
            ema = _exponential_moving_average(prices, span)
            indicators[f'ema_{span}'] = ema

    # MACD (Moving Average Convergence Divergence)
    if prices.size(1) >= 26:
        ema_12 = _exponential_moving_average(prices, 12)
        ema_26 = _exponential_moving_average(prices, 26)
        macd = ema_12 - ema_26
        indicators['macd'] = macd

        if prices.size(1) >= 35:
            signal = _exponential_moving_average(macd, 9)
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = macd - signal

    # RSI (Relative Strength Index)
    if prices.size(1) >= 15:
        rsi = _compute_rsi(prices, 14)
        indicators['rsi_14'] = rsi

    # Bollinger Bands
    if prices.size(1) >= 20:
        sma_20 = _rolling_mean(prices, 20)
        std_20 = _rolling_std(prices, 20)
        indicators['bollinger_upper'] = sma_20 + 2 * std_20
        indicators['bollinger_lower'] = sma_20 - 2 * std_20
        indicators['bollinger_width'] = 4 * std_20 / (sma_20 + 1e-8)
        indicators['bollinger_position'] = (prices - sma_20) / (2 * std_20 + 1e-8)

    # Momentum
    for period in [5, 10, 20]:
        if prices.size(1) > period:
            momentum = prices[:, period:] - prices[:, :-period]
            # Pad to match length
            padding = torch.zeros(prices.size(0), period, device=prices.device)
            indicators[f'momentum_{period}'] = torch.cat([padding, momentum], dim=1)

    # ATR (Average True Range) - requires high, low
    if high is not None and low is not None:
        if high.dim() == 1:
            high = high.unsqueeze(0)
            low = low.unsqueeze(0)

        true_range = torch.maximum(
            high[:, 1:] - low[:, 1:],
            torch.maximum(
                (high[:, 1:] - prices[:, :-1]).abs(),
                (low[:, 1:] - prices[:, :-1]).abs()
            )
        )
        atr = _rolling_mean(true_range, 14)
        # Pad
        padding = torch.zeros(prices.size(0), 1, device=prices.device)
        indicators['atr_14'] = torch.cat([padding, atr], dim=1)

    # Volume indicators
    if volume is not None:
        if volume.dim() == 1:
            volume = volume.unsqueeze(0)

        indicators['volume_sma_20'] = _rolling_mean(volume, 20)
        indicators['volume_ratio'] = volume / (indicators['volume_sma_20'] + 1e-8)

    return indicators


def _rolling_mean(x: Tensor, window: int) -> Tensor:
    """Compute rolling mean."""
    if x.size(1) < window:
        return torch.full_like(x, float('nan'))

    # Use unfold for efficient computation
    x_unfold = x.unfold(dimension=1, size=window, step=1)
    means = x_unfold.mean(dim=-1)

    # Pad beginning with NaN or first valid value
    padding = means[:, :1].expand(-1, window - 1)
    return torch.cat([padding, means], dim=1)


def _rolling_std(x: Tensor, window: int) -> Tensor:
    """Compute rolling standard deviation."""
    if x.size(1) < window:
        return torch.full_like(x, float('nan'))

    x_unfold = x.unfold(dimension=1, size=window, step=1)
    stds = x_unfold.std(dim=-1)

    padding = stds[:, :1].expand(-1, window - 1)
    return torch.cat([padding, stds], dim=1)


def _exponential_moving_average(x: Tensor, span: int) -> Tensor:
    """Compute exponential moving average."""
    alpha = 2 / (span + 1)

    ema = torch.zeros_like(x)
    ema[:, 0] = x[:, 0]

    for t in range(1, x.size(1)):
        ema[:, t] = alpha * x[:, t] + (1 - alpha) * ema[:, t - 1]

    return ema


def _compute_rsi(prices: Tensor, period: int = 14) -> Tensor:
    """Compute Relative Strength Index."""
    deltas = prices.diff(dim=1)

    gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
    losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))

    avg_gain = _rolling_mean(gains, period)
    avg_loss = _rolling_mean(losses, period)

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    # Pad for the first element lost in diff
    padding = torch.full((prices.size(0), 1), 50.0, device=prices.device)
    return torch.cat([padding, rsi], dim=1)


def create_sequences(
    data: Tensor,
    seq_len: int,
    stride: int = 1,
) -> Tensor:
    """
    Create overlapping sequences for time series modeling.

    Args:
        data: Input tensor (batch, total_len, features) or (total_len, features).
        seq_len: Length of each sequence.
        stride: Step between sequences.

    Returns:
        Sequences tensor (n_sequences, seq_len, features).
    """
    if data.dim() == 2:
        data = data.unsqueeze(0)

    batch, total_len, features = data.shape

    sequences = []
    for start in range(0, total_len - seq_len + 1, stride):
        seq = data[:, start:start + seq_len, :]
        sequences.append(seq)

    return torch.cat(sequences, dim=0)
