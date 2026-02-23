"""
Factor Model Layer for Physics-Informed Networks.

Implements factor model structure as a differentiable layer:
    R_i = α_i + Σ_j β_{ij} · F_j + ε_i

Factor models are fundamental to asset pricing and portfolio management:
1. CAPM: Single market factor
2. Fama-French: Market, size (SMB), value (HML) factors
3. APT: Arbitrary number of factors

The layer learns:
- Factor exposures (betas) - how sensitive each asset is to each factor
- Alphas - excess returns not explained by factors
- Idiosyncratic volatilities - asset-specific risk

Key Constraints:
- Factor orthogonality: ε should be uncorrelated with F
- No arbitrage: Expected α should be zero (approximately)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class FactorModelLayer(nn.Module):
    """
    Factor Model Layer enforcing linear factor structure.

    This layer ensures predicted returns follow factor model dynamics:
        R_i = α_i + β_i · F + ε_i

    where β_i can be:
    - Static: Constant loadings (traditional factor model)
    - Dynamic: Time-varying loadings (conditional factor model)
    - Characteristic-based: Loadings depend on asset characteristics

    The layer can be used for:
    1. Return prediction with factor structure
    2. Risk decomposition (systematic vs. idiosyncratic)
    3. Alpha extraction
    """

    def __init__(
        self,
        n_assets: int,
        n_factors: int,
        hidden_dim: int = 64,
        dynamic_betas: bool = False,
        characteristic_dim: Optional[int] = None,
    ):
        """
        Initialize factor model layer.

        Args:
            n_assets: Number of assets.
            n_factors: Number of factors.
            hidden_dim: Hidden dimension for beta networks.
            dynamic_betas: If True, betas depend on time-varying inputs.
            characteristic_dim: Dimension of asset characteristics
                               (required if dynamic_betas=True).
        """
        super().__init__()

        self.n_assets = n_assets
        self.n_factors = n_factors
        self.dynamic_betas = dynamic_betas

        if dynamic_betas:
            if characteristic_dim is None:
                raise ValueError("characteristic_dim required for dynamic betas")

            # Network that maps characteristics to betas
            self.beta_network = nn.Sequential(
                nn.Linear(characteristic_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, n_assets * n_factors),
            )

            # Network for dynamic alphas
            self.alpha_network = nn.Sequential(
                nn.Linear(characteristic_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, n_assets),
            )
        else:
            # Static betas and alphas (learnable parameters)
            # Initialize betas near 1 for market factor, near 0 for others
            beta_init = torch.zeros(n_assets, n_factors)
            beta_init[:, 0] = 1.0  # Market beta = 1 initially
            self.betas = nn.Parameter(beta_init)

            # Initialize alphas to zero (no expected alpha)
            self.alphas = nn.Parameter(torch.zeros(n_assets))

    def forward(
        self,
        factor_returns: Tensor,
        characteristics: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute factor model returns and exposures.

        Args:
            factor_returns: Factor returns of shape (batch, n_factors).
            characteristics: Asset characteristics of shape (batch, n_assets, char_dim).
                            Required if dynamic_betas=True.

        Returns:
            Tuple of:
                - predicted_returns: Shape (batch, n_assets)
                - betas: Shape (batch, n_assets, n_factors) or (n_assets, n_factors)
                - alphas: Shape (batch, n_assets) or (n_assets,)
        """
        batch_size = factor_returns.shape[0]

        if self.dynamic_betas:
            if characteristics is None:
                raise ValueError("characteristics required for dynamic betas")

            # Get time-varying betas from characteristics
            # characteristics: (batch, n_assets, char_dim) -> flatten -> network
            char_flat = characteristics.view(batch_size * self.n_assets, -1)
            betas_flat = self.beta_network(char_flat)
            betas = betas_flat.view(batch_size, self.n_assets, self.n_factors)

            # Get time-varying alphas
            alphas = self.alpha_network(characteristics.mean(dim=1))  # (batch, n_assets)
        else:
            # Use static parameters
            betas = self.betas.unsqueeze(0).expand(batch_size, -1, -1)
            alphas = self.alphas.unsqueeze(0).expand(batch_size, -1)

        # Compute factor contribution: β · F
        # betas: (batch, n_assets, n_factors)
        # factor_returns: (batch, n_factors)
        # Result: (batch, n_assets)
        factor_contribution = torch.einsum('baf,bf->ba', betas, factor_returns)

        # Total predicted return: α + β · F
        predicted_returns = alphas + factor_contribution

        return predicted_returns, betas, alphas

    def factor_residual_loss(
        self,
        actual_returns: Tensor,
        factor_returns: Tensor,
        characteristics: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute loss for factor model consistency.

        The residual (ε = R - α - β·F) should be:
        1. Zero-mean
        2. Uncorrelated with factors
        3. Have reasonable variance (not too large)

        Args:
            actual_returns: Actual returns of shape (batch, n_assets).
            factor_returns: Factor returns of shape (batch, n_factors).
            characteristics: Asset characteristics (optional).

        Returns:
            Scalar loss value.
        """
        predicted, betas, alphas = self.forward(factor_returns, characteristics)

        # Compute residuals
        residuals = actual_returns - predicted  # (batch, n_assets)

        # Loss 1: Residuals should be zero-mean (across batch)
        mean_residual = residuals.mean(dim=0)
        zero_mean_loss = mean_residual.pow(2).mean()

        # Loss 2: Residuals should be uncorrelated with factors
        # Compute correlation: E[ε·F] should be 0
        # residuals: (batch, n_assets), factor_returns: (batch, n_factors)
        # Outer product and sum over batch
        correlation = torch.einsum('ba,bf->af', residuals, factor_returns) / residuals.shape[0]
        orthogonality_loss = correlation.pow(2).mean()

        # Combined loss
        total_loss = zero_mean_loss + orthogonality_loss

        return total_loss

    def decompose_risk(
        self,
        factor_covariance: Tensor,
        betas: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Decompose total variance into systematic and idiosyncratic.

        Var(R_i) = β_i' Σ_F β_i + Var(ε_i)

        where Σ_F is the factor covariance matrix.

        Args:
            factor_covariance: Factor covariance matrix (n_factors, n_factors).
            betas: Optional beta override. If None, uses learned betas.

        Returns:
            Tuple of (systematic_variance, total_variance_contribution).
        """
        if betas is None:
            if self.dynamic_betas:
                raise ValueError("betas required for dynamic model")
            betas = self.betas

        # Systematic variance: β' Σ_F β
        # For each asset i: betas[i] @ factor_cov @ betas[i]
        systematic_var = torch.einsum(
            'af,fg,ag->a',
            betas, factor_covariance, betas
        )

        return systematic_var

    def get_factor_exposures(self) -> Tensor:
        """Get current factor exposures (betas)."""
        if self.dynamic_betas:
            raise ValueError("Betas are dynamic; provide characteristics to forward()")
        return self.betas.detach()


class FamaFrenchLayer(FactorModelLayer):
    """
    Specialized layer for Fama-French three-factor model.

    Factors:
    1. Market (Rm - Rf): Excess market return
    2. SMB (Small Minus Big): Size factor
    3. HML (High Minus Low): Value factor

    This provides interpretable factor loadings for equity returns.
    """

    def __init__(
        self,
        n_assets: int,
        hidden_dim: int = 64,
        dynamic_betas: bool = False,
        characteristic_dim: Optional[int] = None,
    ):
        """
        Initialize Fama-French layer.

        Args:
            n_assets: Number of assets.
            hidden_dim: Hidden dimension for networks.
            dynamic_betas: If True, betas depend on characteristics.
            characteristic_dim: Dimension of characteristics.
        """
        super().__init__(
            n_assets=n_assets,
            n_factors=3,  # MKT, SMB, HML
            hidden_dim=hidden_dim,
            dynamic_betas=dynamic_betas,
            characteristic_dim=characteristic_dim,
        )

        self.factor_names = ['MKT', 'SMB', 'HML']

    def get_named_exposures(self) -> dict:
        """Get factor exposures with factor names."""
        betas = self.get_factor_exposures()
        return {
            name: betas[:, i].detach()
            for i, name in enumerate(self.factor_names)
        }


class APTLayer(nn.Module):
    """
    Arbitrage Pricing Theory (APT) Layer.

    More general than Fama-French; factors can be:
    - Statistical (PCA-derived)
    - Macroeconomic (GDP, inflation, etc.)
    - Fundamental (earnings, book-to-market, etc.)

    Includes APT no-arbitrage constraint: E[R] = λ₀ + β'λ
    where λ is the factor risk premium vector.
    """

    def __init__(
        self,
        n_assets: int,
        n_factors: int,
        hidden_dim: int = 64,
        enforce_apt: bool = True,
    ):
        """
        Initialize APT layer.

        Args:
            n_assets: Number of assets.
            n_factors: Number of factors.
            hidden_dim: Hidden dimension.
            enforce_apt: If True, add APT pricing constraint.
        """
        super().__init__()

        self.n_assets = n_assets
        self.n_factors = n_factors
        self.enforce_apt = enforce_apt

        # Factor loadings
        self.betas = nn.Parameter(torch.randn(n_assets, n_factors) * 0.1)

        # Factor risk premiums (λ)
        self.risk_premiums = nn.Parameter(torch.zeros(n_factors))

        # Risk-free rate (λ₀)
        self.rf = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        factor_returns: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute APT expected returns.

        Args:
            factor_returns: Factor realizations (batch, n_factors).

        Returns:
            Tuple of (predicted_returns, expected_returns_apt).
        """
        batch_size = factor_returns.shape[0]

        # Realized returns: β · F
        factor_contribution = factor_returns @ self.betas.T  # (batch, n_assets)

        # APT expected returns: rf + β · λ
        expected_returns = self.rf + self.betas @ self.risk_premiums

        return factor_contribution, expected_returns

    def apt_pricing_loss(
        self,
        actual_mean_returns: Tensor,
    ) -> Tensor:
        """
        APT pricing constraint: E[R_i] = rf + β_i · λ

        This should hold approximately for all assets.

        Args:
            actual_mean_returns: Sample mean returns (n_assets,).

        Returns:
            Pricing error loss.
        """
        expected = self.rf + self.betas @ self.risk_premiums
        pricing_error = actual_mean_returns - expected
        return pricing_error.pow(2).mean()
