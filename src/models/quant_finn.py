"""
Quant-FINN: Full Quantitative Finance-Informed Neural Network.

This is the main model integrating all components:
- Feature encoders (temporal, Fourier, etc.)
- Physics-informed layers (GARCH, factor model, portfolio)
- Regime detection
- Multiple output heads

Architecture:
    Inputs (prices, technicals, fundamentals)
        |
        v
    Feature Encoding (LSTM, Fourier, fusion)
        |
        v
    Regime Detection (classifier)
        |
        v
    Physics-Informed Core (GARCH, factor model, portfolio optimization)
        |
        v
    Output Heads (returns, volatility, positions, confidence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from src.models.encoders import (
    FourierFeatureEncoder,
    TemporalEncoder,
    FeatureFusion,
)
from src.models.layers import (
    GARCHLayer,
    GARCHRegimeLayer,
    FactorModelLayer,
    MeanVarianceLayer,
)


@dataclass
class QuantFINNConfig:
    """Configuration for Quant-FINN model."""

    # Input dimensions
    n_assets: int = 10
    n_factors: int = 3
    seq_len: int = 60
    price_features: int = 5  # OHLCV
    technical_features: int = 10
    fundamental_features: int = 5

    # Encoder dimensions
    hidden_dim: int = 64
    encoder_layers: int = 2
    n_fourier_frequencies: int = 10

    # Regime detection
    n_regimes: int = 3

    # Output options
    predict_returns: bool = True
    predict_volatility: bool = True
    predict_positions: bool = True
    predict_confidence: bool = True

    # Physics constraints
    use_garch: bool = True
    use_factor_model: bool = True
    use_portfolio_opt: bool = True

    # Training
    dropout: float = 0.1
    risk_aversion: float = 1.0


class RegimeClassifier(nn.Module):
    """
    Market Regime Classifier.

    Identifies market regimes (e.g., bull, bear, sideways) from
    encoded features. Regime probabilities are used to weight
    regime-specific model components.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_regimes: int,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_regimes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Classify market regime.

        Args:
            x: Encoded features (batch, input_dim).

        Returns:
            Regime probabilities (batch, n_regimes).
        """
        logits = self.classifier(x)
        return F.softmax(logits, dim=-1)


class QuantFINN(nn.Module):
    """
    Quantitative Finance-Informed Neural Network.

    The main model combining:
    1. Feature encoding: Transform raw inputs to learned representations
    2. Regime detection: Identify market state
    3. Physics-informed layers: GARCH, factor model, portfolio optimization
    4. Output heads: Returns, volatility, positions, confidence

    The model enforces financial constraints through:
    - GARCH volatility dynamics
    - Factor model structure
    - Mean-variance optimality conditions
    - Regime-dependent parameters
    """

    def __init__(self, config: QuantFINNConfig):
        """
        Initialize Quant-FINN model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # === Feature Encoders ===

        # Temporal encoder for price sequences
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.price_features,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            n_layers=config.encoder_layers,
            dropout=config.dropout,
        )

        # Fourier encoder for technical indicators
        self.fourier_encoder = FourierFeatureEncoder(
            input_dim=config.technical_features,
            output_dim=config.hidden_dim,
            n_frequencies=config.n_fourier_frequencies,
        )

        # Simple encoder for fundamental data
        self.fundamental_encoder = nn.Sequential(
            nn.Linear(config.fundamental_features * config.n_assets, config.hidden_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )

        # Feature fusion
        self.fusion = FeatureFusion(
            input_dims=[config.hidden_dim, config.hidden_dim, config.hidden_dim],
            output_dim=config.hidden_dim,
            fusion_type='gated',
        )

        # === Regime Detection ===
        self.regime_classifier = RegimeClassifier(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            n_regimes=config.n_regimes,
        )

        # === Physics-Informed Layers ===

        if config.use_garch:
            self.garch = GARCHRegimeLayer(n_regimes=config.n_regimes)

        if config.use_factor_model:
            self.factor_model = FactorModelLayer(
                n_assets=config.n_assets,
                n_factors=config.n_factors,
                hidden_dim=config.hidden_dim,
                dynamic_betas=True,
                characteristic_dim=config.fundamental_features,
            )

        if config.use_portfolio_opt:
            self.portfolio_optimizer = MeanVarianceLayer(
                n_assets=config.n_assets,
                risk_aversion=config.risk_aversion,
                long_only=True,
            )

        # === Output Heads ===

        # Shared hidden layer before heads
        self.output_hidden = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout),
        )

        if config.predict_returns:
            self.return_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(config.hidden_dim // 2, config.n_assets),
            )

        if config.predict_volatility:
            self.volatility_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(config.hidden_dim // 2, config.n_assets),
                nn.Softplus(),  # Ensure positive volatility
            )

        if config.predict_positions:
            self.position_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(config.hidden_dim // 2, config.n_assets),
                nn.Tanh(),  # Positions in [-1, 1]
            )

        if config.predict_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid(),  # Confidence in [0, 1]
            )

    def encode_features(
        self,
        price_data: Tensor,
        technical_data: Tensor,
        fundamental_data: Tensor,
    ) -> Tensor:
        """
        Encode all input features into unified representation.

        Args:
            price_data: Price sequences (batch, seq_len, price_features).
            technical_data: Technical indicators (batch, technical_features).
            fundamental_data: Fundamentals (batch, n_assets, fundamental_features).

        Returns:
            Fused features (batch, hidden_dim).
        """
        # Temporal encoding of price sequences
        temporal_features = self.temporal_encoder(price_data)

        # Fourier encoding of technical indicators
        fourier_features = self.fourier_encoder(technical_data)

        # Fundamental encoding
        fundamental_flat = fundamental_data.flatten(start_dim=1)
        fundamental_features = self.fundamental_encoder(fundamental_flat)

        # Fuse all features
        fused = self.fusion(temporal_features, fourier_features, fundamental_features)

        return fused

    def forward(
        self,
        price_data: Tensor,
        technical_data: Tensor,
        fundamental_data: Tensor,
        factor_returns: Optional[Tensor] = None,
        returns_history: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through Quant-FINN.

        Args:
            price_data: Price sequences (batch, seq_len, price_features).
            technical_data: Technical indicators (batch, technical_features).
            fundamental_data: Fundamentals (batch, n_assets, fundamental_features).
            factor_returns: Factor returns (batch, n_factors). Required for factor model.
            returns_history: Historical returns (batch, hist_len). Required for GARCH.

        Returns:
            Dict with predictions:
                - 'returns': Predicted returns (batch, n_assets)
                - 'volatility': Predicted volatility (batch, n_assets)
                - 'positions': Suggested positions (batch, n_assets)
                - 'optimal_positions': Mean-variance optimal positions
                - 'confidence': Model confidence (batch, 1)
                - 'regime_probs': Regime probabilities (batch, n_regimes)
                - 'betas': Factor exposures (batch, n_assets, n_factors)
        """
        outputs = {}

        # Encode features
        features = self.encode_features(price_data, technical_data, fundamental_data)

        # Detect regime
        regime_probs = self.regime_classifier(features)
        outputs['regime_probs'] = regime_probs

        # Process through hidden layer
        hidden = self.output_hidden(features)

        # === Output Heads ===

        if self.config.predict_returns:
            outputs['returns'] = self.return_head(hidden)

        if self.config.predict_volatility:
            outputs['volatility'] = self.volatility_head(hidden)

        if self.config.predict_positions:
            outputs['positions'] = self.position_head(hidden)

        if self.config.predict_confidence:
            outputs['confidence'] = self.confidence_head(hidden)

        # === Physics-Informed Components ===

        # GARCH volatility forecast
        if self.config.use_garch and returns_history is not None:
            garch_variance = self.garch(returns_history, regime_probs)
            outputs['garch_volatility'] = torch.sqrt(garch_variance[:, -1] + 1e-8)

        # Factor model
        if self.config.use_factor_model and factor_returns is not None:
            factor_returns_pred, betas, alphas = self.factor_model(
                factor_returns, fundamental_data
            )
            outputs['factor_returns'] = factor_returns_pred
            outputs['betas'] = betas
            outputs['alphas'] = alphas

        # Portfolio optimization
        if self.config.use_portfolio_opt and 'returns' in outputs and 'volatility' in outputs:
            # Create simple covariance estimate from volatilities
            # (In practice, you'd use a more sophisticated estimate)
            vol = outputs['volatility']
            cov = torch.diag_embed(vol.pow(2))

            optimal_positions = self.portfolio_optimizer(outputs['returns'], cov)
            outputs['optimal_positions'] = optimal_positions

        return outputs

    def compute_physics_loss(
        self,
        outputs: Dict[str, Tensor],
        actual_returns: Tensor,
        actual_volatility: Optional[Tensor] = None,
        factor_returns: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute physics-informed loss components.

        Args:
            outputs: Model outputs from forward pass.
            actual_returns: Actual observed returns.
            actual_volatility: Actual realized volatility (optional).
            factor_returns: Factor returns (optional).

        Returns:
            Dict of loss components.
        """
        losses = {}

        # Prediction losses
        if 'returns' in outputs:
            losses['returns'] = F.mse_loss(outputs['returns'], actual_returns)

        if 'volatility' in outputs and actual_volatility is not None:
            losses['volatility'] = F.mse_loss(outputs['volatility'], actual_volatility)

        # GARCH consistency loss
        if 'garch_volatility' in outputs and actual_volatility is not None:
            losses['garch'] = F.mse_loss(
                outputs['garch_volatility'],
                actual_volatility.mean(dim=-1)
            )

        # Factor model structure loss
        if self.config.use_factor_model and 'betas' in outputs and factor_returns is not None:
            losses['factor'] = self.factor_model.factor_residual_loss(
                actual_returns, factor_returns, None  # characteristics
            )

        # Portfolio optimality loss
        if 'positions' in outputs and 'optimal_positions' in outputs:
            losses['portfolio'] = F.mse_loss(
                outputs['positions'],
                outputs['optimal_positions']
            )

        return losses

    def get_regime_parameters(self) -> Dict:
        """Get current regime-specific parameters."""
        params = {}

        if self.config.use_garch:
            params['garch'] = self.garch.get_regime_parameters()

        return params


class SimpleQuantFINN(nn.Module):
    """
    Simplified Quant-FINN for quick experiments.

    A lighter version without all the bells and whistles,
    suitable for:
    - Prototyping
    - Single-asset trading
    - Educational purposes
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        n_layers: int = 4,
        output_dim: int = 1,
        use_garch: bool = True,
    ):
        super().__init__()

        # Simple MLP backbone
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.return_head = nn.Linear(hidden_dim, output_dim)
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),
        )

        # GARCH layer
        if use_garch:
            self.garch = GARCHLayer(learnable=True)
        else:
            self.garch = None

    def forward(
        self,
        x: Tensor,
        returns_history: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim).
            returns_history: Historical returns for GARCH (batch, seq_len).

        Returns:
            Dict with 'returns', 'volatility', and optionally 'garch_volatility'.
        """
        features = self.backbone(x)

        outputs = {
            'returns': self.return_head(features),
            'volatility': self.volatility_head(features),
        }

        if self.garch is not None and returns_history is not None:
            garch_var = self.garch(returns_history)
            outputs['garch_volatility'] = torch.sqrt(garch_var[:, -1:] + 1e-8)

        return outputs
