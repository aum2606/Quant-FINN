"""
Portfolio Optimization Layers for Physics-Informed Networks.

Implements portfolio optimization as differentiable layers:
1. Mean-Variance: Classic Markowitz optimization
2. Risk Parity: Equal risk contribution
3. Maximum Sharpe: Tangency portfolio

These layers enforce portfolio optimality conditions as constraints,
ensuring network outputs are economically sensible portfolios.

Key Constraints:
- Weights sum to 1 (fully invested)
- Optional: Long-only (weights ≥ 0)
- Optional: Position limits
- Optimality conditions (first-order conditions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class MeanVarianceLayer(nn.Module):
    """
    Mean-Variance Portfolio Optimization Layer.

    Implements Markowitz mean-variance optimization:
        maximize: μ'w - (λ/2) w'Σw
        subject to: 1'w = 1, w ≥ 0 (optional)

    The closed-form solution (unconstrained) is:
        w* = (1/λ) Σ⁻¹ (μ - ν·1)

    where ν is chosen to satisfy the budget constraint.

    This layer can:
    1. Compute optimal weights given μ and Σ
    2. Enforce optimality as a soft constraint on network outputs
    3. Handle constraints via projection or penalties
    """

    def __init__(
        self,
        n_assets: int,
        risk_aversion: float = 1.0,
        long_only: bool = True,
        max_position: float = 0.25,
    ):
        """
        Initialize mean-variance layer.

        Args:
            n_assets: Number of assets.
            risk_aversion: Risk aversion parameter λ (higher = more conservative).
            long_only: If True, enforce non-negative weights.
            max_position: Maximum weight for any single asset.
        """
        super().__init__()

        self.n_assets = n_assets
        self.long_only = long_only
        self.max_position = max_position

        # Risk aversion can be learnable
        self.log_risk_aversion = nn.Parameter(torch.log(torch.tensor(risk_aversion)))

    @property
    def risk_aversion(self) -> Tensor:
        """Risk aversion parameter (always positive)."""
        return torch.exp(self.log_risk_aversion)

    def forward(
        self,
        expected_returns: Tensor,
        covariance: Tensor,
    ) -> Tensor:
        """
        Compute optimal portfolio weights.

        For numerical stability, we use a regularized inverse.

        Args:
            expected_returns: Expected returns (batch, n_assets).
            covariance: Covariance matrix (batch, n_assets, n_assets)
                       or (n_assets, n_assets) for shared covariance.

        Returns:
            Optimal weights (batch, n_assets).
        """
        batch_size = expected_returns.shape[0]

        # Handle shared covariance
        if covariance.dim() == 2:
            covariance = covariance.unsqueeze(0).expand(batch_size, -1, -1)

        # Regularize covariance for numerical stability
        reg = 1e-6 * torch.eye(self.n_assets, device=covariance.device)
        cov_reg = covariance + reg

        # Compute inverse
        # Using solve for numerical stability
        try:
            cov_inv = torch.linalg.inv(cov_reg)
        except:
            # Fallback to pseudo-inverse
            cov_inv = torch.linalg.pinv(cov_reg)

        # Unconstrained optimal: w* = (1/λ) Σ⁻¹ μ (ignoring budget constraint for now)
        ones = torch.ones(self.n_assets, device=expected_returns.device)

        # Compute intermediate quantities
        Sigma_inv_mu = torch.einsum('bij,bj->bi', cov_inv, expected_returns)
        Sigma_inv_one = torch.einsum('bij,j->bi', cov_inv, ones)

        # Find ν to satisfy budget constraint: 1'w = 1
        # w = (1/λ)(Σ⁻¹μ - ν·Σ⁻¹·1)
        # 1'w = 1 => ν = (1'Σ⁻¹μ - λ) / (1'Σ⁻¹·1)
        one_Sigma_inv_mu = torch.einsum('bi,i->b', Sigma_inv_mu, ones)
        one_Sigma_inv_one = torch.einsum('bi,i->b', Sigma_inv_one, ones)

        nu = (one_Sigma_inv_mu - self.risk_aversion) / (one_Sigma_inv_one + 1e-8)

        # Optimal weights
        weights = (Sigma_inv_mu - nu.unsqueeze(-1) * Sigma_inv_one) / self.risk_aversion

        # Apply constraints
        weights = self._apply_constraints(weights)

        return weights

    def _apply_constraints(self, weights: Tensor) -> Tensor:
        """Apply portfolio constraints to weights."""
        # Long-only constraint
        if self.long_only:
            weights = F.relu(weights)

        # Normalize to sum to 1
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Position limits
        if self.max_position < 1.0:
            weights = torch.clamp(weights, max=self.max_position)
            # Renormalize after clamping
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return weights

    def optimality_loss(
        self,
        weights: Tensor,
        expected_returns: Tensor,
        covariance: Tensor,
    ) -> Tensor:
        """
        Compute loss for deviation from optimality conditions.

        First-order conditions: μ - λΣw = ν·1 (same ν for all assets)

        The gradient μ - λΣw should be constant across assets.

        Args:
            weights: Current weights (batch, n_assets).
            expected_returns: Expected returns (batch, n_assets).
            covariance: Covariance matrix.

        Returns:
            Optimality loss (deviation from FOC).
        """
        batch_size = weights.shape[0]

        if covariance.dim() == 2:
            covariance = covariance.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute gradient of objective: μ - λΣw
        Sigma_w = torch.einsum('bij,bj->bi', covariance, weights)
        gradient = expected_returns - self.risk_aversion * Sigma_w

        # For optimality, gradient should be constant (= ν)
        gradient_mean = gradient.mean(dim=-1, keepdim=True)
        deviation = gradient - gradient_mean

        return deviation.pow(2).mean()

    def compute_portfolio_stats(
        self,
        weights: Tensor,
        expected_returns: Tensor,
        covariance: Tensor,
    ) -> dict:
        """
        Compute portfolio statistics.

        Args:
            weights: Portfolio weights.
            expected_returns: Expected returns.
            covariance: Covariance matrix.

        Returns:
            Dict with expected return, volatility, Sharpe ratio.
        """
        batch_size = weights.shape[0]

        if covariance.dim() == 2:
            covariance = covariance.unsqueeze(0).expand(batch_size, -1, -1)

        # Portfolio expected return: μ'w
        port_return = torch.einsum('bi,bi->b', weights, expected_returns)

        # Portfolio variance: w'Σw
        Sigma_w = torch.einsum('bij,bj->bi', covariance, weights)
        port_variance = torch.einsum('bi,bi->b', weights, Sigma_w)
        port_volatility = torch.sqrt(port_variance + 1e-8)

        # Sharpe ratio (assuming rf = 0)
        sharpe = port_return / (port_volatility + 1e-8)

        return {
            'expected_return': port_return,
            'volatility': port_volatility,
            'variance': port_variance,
            'sharpe_ratio': sharpe,
        }


class RiskParityLayer(nn.Module):
    """
    Risk Parity Portfolio Layer.

    Risk parity allocates capital so each asset contributes equally
    to total portfolio risk:
        w_i * (Σw)_i / (w'Σw) = 1/n

    This is more robust than mean-variance when expected returns
    are uncertain, as it only uses covariance information.
    """

    def __init__(
        self,
        n_assets: int,
        long_only: bool = True,
        max_iterations: int = 50,
    ):
        """
        Initialize risk parity layer.

        Args:
            n_assets: Number of assets.
            long_only: If True, enforce non-negative weights.
            max_iterations: Maximum iterations for optimization.
        """
        super().__init__()

        self.n_assets = n_assets
        self.long_only = long_only
        self.max_iterations = max_iterations

    def forward(
        self,
        covariance: Tensor,
        target_risk_contributions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute risk parity weights.

        Uses iterative algorithm to find weights where each asset
        has equal (or target) risk contribution.

        Args:
            covariance: Covariance matrix (batch, n_assets, n_assets)
                       or (n_assets, n_assets).
            target_risk_contributions: Target risk contributions per asset.
                                      Defaults to 1/n for all assets.

        Returns:
            Risk parity weights (batch, n_assets).
        """
        if covariance.dim() == 2:
            covariance = covariance.unsqueeze(0)

        batch_size = covariance.shape[0]
        device = covariance.device

        # Default: equal risk contribution
        if target_risk_contributions is None:
            target_risk_contributions = torch.ones(
                batch_size, self.n_assets, device=device
            ) / self.n_assets

        # Initialize with equal weights
        weights = torch.ones(batch_size, self.n_assets, device=device) / self.n_assets

        # Iterative optimization
        for _ in range(self.max_iterations):
            # Compute marginal risk contributions
            Sigma_w = torch.einsum('bij,bj->bi', covariance, weights)
            port_variance = torch.einsum('bi,bi->b', weights, Sigma_w)
            port_vol = torch.sqrt(port_variance + 1e-8).unsqueeze(-1)

            # Risk contribution: w_i * (Σw)_i / σ_p
            risk_contrib = weights * Sigma_w / (port_vol + 1e-8)

            # Total risk contribution (should sum to port_vol)
            total_risk = risk_contrib.sum(dim=-1, keepdim=True)

            # Normalized risk contribution
            risk_contrib_norm = risk_contrib / (total_risk + 1e-8)

            # Update weights to move toward target
            # Simple gradient step
            gradient = target_risk_contributions - risk_contrib_norm
            weights = weights + 0.1 * gradient * weights

            # Ensure positivity and normalization
            if self.long_only:
                weights = F.relu(weights)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return weights

    def risk_contribution(
        self,
        weights: Tensor,
        covariance: Tensor,
    ) -> Tensor:
        """
        Compute risk contribution of each asset.

        Risk contribution of asset i:
            RC_i = w_i * (Σw)_i / σ_p

        Sum of all RC_i equals portfolio volatility.

        Args:
            weights: Portfolio weights (batch, n_assets).
            covariance: Covariance matrix.

        Returns:
            Risk contributions (batch, n_assets).
        """
        if covariance.dim() == 2:
            covariance = covariance.unsqueeze(0)

        Sigma_w = torch.einsum('bij,bj->bi', covariance, weights)
        port_variance = torch.einsum('bi,bi->b', weights, Sigma_w)
        port_vol = torch.sqrt(port_variance + 1e-8).unsqueeze(-1)

        risk_contrib = weights * Sigma_w / (port_vol + 1e-8)

        return risk_contrib

    def risk_parity_loss(
        self,
        weights: Tensor,
        covariance: Tensor,
    ) -> Tensor:
        """
        Compute loss for deviation from risk parity.

        All normalized risk contributions should be equal (1/n).

        Args:
            weights: Portfolio weights.
            covariance: Covariance matrix.

        Returns:
            Risk parity deviation loss.
        """
        risk_contrib = self.risk_contribution(weights, covariance)
        total_risk = risk_contrib.sum(dim=-1, keepdim=True)
        risk_contrib_norm = risk_contrib / (total_risk + 1e-8)

        # Target: 1/n for all assets
        target = 1.0 / self.n_assets
        deviation = risk_contrib_norm - target

        return deviation.pow(2).mean()


class MaxSharpeLayer(nn.Module):
    """
    Maximum Sharpe Ratio (Tangency) Portfolio Layer.

    Finds the portfolio with maximum Sharpe ratio:
        maximize: (μ'w - rf) / sqrt(w'Σw)

    This is the tangency portfolio - the optimal risky portfolio
    when a risk-free asset is available.
    """

    def __init__(
        self,
        n_assets: int,
        rf: float = 0.0,
        long_only: bool = True,
    ):
        """
        Initialize max Sharpe layer.

        Args:
            n_assets: Number of assets.
            rf: Risk-free rate.
            long_only: If True, enforce non-negative weights.
        """
        super().__init__()

        self.n_assets = n_assets
        self.long_only = long_only
        self.register_buffer('rf', torch.tensor(rf))

    def forward(
        self,
        expected_returns: Tensor,
        covariance: Tensor,
    ) -> Tensor:
        """
        Compute maximum Sharpe ratio portfolio.

        Closed-form solution:
            w* = Σ⁻¹(μ - rf·1) / (1'Σ⁻¹(μ - rf·1))

        Args:
            expected_returns: Expected returns (batch, n_assets).
            covariance: Covariance matrix.

        Returns:
            Optimal weights (batch, n_assets).
        """
        batch_size = expected_returns.shape[0]

        if covariance.dim() == 2:
            covariance = covariance.unsqueeze(0).expand(batch_size, -1, -1)

        # Excess returns
        excess_returns = expected_returns - self.rf

        # Regularize and invert covariance
        reg = 1e-6 * torch.eye(self.n_assets, device=covariance.device)
        cov_reg = covariance + reg

        try:
            cov_inv = torch.linalg.inv(cov_reg)
        except:
            cov_inv = torch.linalg.pinv(cov_reg)

        # Optimal weights (unnormalized)
        Sigma_inv_excess = torch.einsum('bij,bj->bi', cov_inv, excess_returns)

        # Normalize
        ones = torch.ones(self.n_assets, device=expected_returns.device)
        normalizer = torch.einsum('bi,i->b', Sigma_inv_excess, ones)

        weights = Sigma_inv_excess / (normalizer.unsqueeze(-1) + 1e-8)

        # Apply constraints
        if self.long_only:
            weights = F.relu(weights)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return weights
