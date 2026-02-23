"""
GARCH Volatility Layer for Physics-Informed Networks.

Implements GARCH(1,1) dynamics as a differentiable layer:
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

GARCH captures key properties of financial volatility:
1. Volatility clustering - high volatility tends to persist
2. Mean reversion - volatility reverts to long-term level
3. Asymmetry (with extensions) - negative shocks increase volatility more

The layer parameters (ω, α, β) can be:
- Fixed (known values from estimation)
- Learnable (estimated jointly with other parameters)
- Regime-dependent (different values for different market states)

Key Constraints:
- ω > 0 (positive long-term variance)
- α ≥ 0, β ≥ 0 (non-negative weights)
- α + β < 1 (stationarity condition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class GARCHLayer(nn.Module):
    """
    GARCH(1,1) Layer for volatility modeling.

    This layer enforces GARCH dynamics by computing conditional variance
    following the GARCH(1,1) recursion. Parameters can be fixed or learned.

    The GARCH(1,1) model:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    where:
        - σ²_t: Conditional variance at time t
        - ε²_{t-1}: Squared shock (return - mean)² at t-1
        - ω: Long-run variance weight (intercept)
        - α: Shock impact coefficient (ARCH term)
        - β: Persistence coefficient (GARCH term)

    The unconditional (long-run) variance is: σ² = ω / (1 - α - β)

    Example:
        >>> garch = GARCHLayer(learnable=True)
        >>> returns = torch.randn(32, 100)  # (batch, seq_len)
        >>> variances = garch(returns)
        >>> print(f"Persistence: {garch.persistence:.3f}")
    """

    def __init__(
        self,
        omega: float = 0.0001,
        alpha: float = 0.1,
        beta: float = 0.85,
        learnable: bool = True,
        constrain_stationarity: bool = True,
    ):
        """
        Initialize GARCH layer.

        Args:
            omega: Initial value for ω (long-run variance weight).
            alpha: Initial value for α (ARCH coefficient).
            beta: Initial value for β (GARCH coefficient).
            learnable: If True, parameters are learned during training.
            constrain_stationarity: If True, enforce α + β < 1.
        """
        super().__init__()

        self.learnable = learnable
        self.constrain_stationarity = constrain_stationarity

        if learnable:
            # Use raw parameters that we transform to valid ranges
            # This ensures constraints are satisfied via transformations
            self._omega_raw = nn.Parameter(torch.tensor(self._inverse_softplus(omega)))
            self._alpha_raw = nn.Parameter(torch.tensor(self._inverse_sigmoid(alpha / 0.5)))
            self._beta_raw = nn.Parameter(torch.tensor(self._inverse_sigmoid(beta / 0.99)))
        else:
            self.register_buffer('_omega', torch.tensor(omega))
            self.register_buffer('_alpha', torch.tensor(alpha))
            self.register_buffer('_beta', torch.tensor(beta))

    @staticmethod
    def _inverse_softplus(x: float) -> float:
        """Inverse of softplus for initialization."""
        return x + torch.log(torch.exp(torch.tensor(x)) - 1).item() if x > 0 else 0.0

    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        """Inverse of sigmoid for initialization."""
        x = max(min(x, 0.999), 0.001)  # Clip to valid range
        return torch.log(torch.tensor(x / (1 - x))).item()

    @property
    def omega(self) -> Tensor:
        """ω parameter (long-run variance weight). Always positive."""
        if self.learnable:
            # Softplus ensures positivity, scale for typical variance magnitudes
            return F.softplus(self._omega_raw) * 0.0001
        return self._omega

    @property
    def alpha(self) -> Tensor:
        """α parameter (ARCH coefficient). Range: [0, 0.5]."""
        if self.learnable:
            # Sigmoid * 0.5 keeps α in reasonable range
            return torch.sigmoid(self._alpha_raw) * 0.5
        return self._alpha

    @property
    def beta(self) -> Tensor:
        """β parameter (GARCH coefficient). Range: [0, 0.99]."""
        if self.learnable:
            if self.constrain_stationarity:
                # Ensure α + β < 1 for stationarity
                max_beta = 0.99 - self.alpha.detach()
                return torch.sigmoid(self._beta_raw) * max_beta
            else:
                return torch.sigmoid(self._beta_raw) * 0.99
        return self._beta

    @property
    def persistence(self) -> Tensor:
        """
        Persistence = α + β.

        Should be < 1 for stationarity.
        Values close to 1 indicate highly persistent volatility.
        """
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> Tensor:
        """
        Long-run (unconditional) variance = ω / (1 - α - β).

        This is the variance the process reverts to over time.
        """
        return self.omega / (1 - self.persistence + 1e-8)

    def forward(
        self,
        returns: Tensor,
        initial_variance: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute GARCH conditional variances for a sequence of returns.

        Args:
            returns: Return series of shape (batch, seq_len).
            initial_variance: Optional initial variance. If None, uses
                              sample variance of returns.

        Returns:
            Conditional variance series of shape (batch, seq_len).
        """
        batch_size, seq_len = returns.shape
        device = returns.device

        # Initialize variance
        if initial_variance is None:
            # Use sample variance as initial estimate
            initial_variance = returns.var(dim=1, keepdim=True)

        # Compute squared shocks (assuming zero mean)
        shocks_sq = returns.pow(2)

        # Initialize variance sequence
        variances = torch.zeros(batch_size, seq_len, device=device)
        variance = initial_variance.squeeze(-1)

        # GARCH recursion
        for t in range(seq_len):
            if t == 0:
                # First period uses initial variance
                variances[:, t] = variance
            else:
                # GARCH(1,1) update
                variance = (
                    self.omega
                    + self.alpha * shocks_sq[:, t - 1]
                    + self.beta * variance
                )
                variances[:, t] = variance

        return variances

    def one_step_ahead(
        self,
        current_variance: Tensor,
        current_shock_sq: Tensor,
    ) -> Tensor:
        """
        Compute one-step-ahead variance forecast.

        Args:
            current_variance: Current conditional variance.
            current_shock_sq: Current squared shock (return²).

        Returns:
            Next period's conditional variance.
        """
        return (
            self.omega
            + self.alpha * current_shock_sq
            + self.beta * current_variance
        )

    def multi_step_forecast(
        self,
        current_variance: Tensor,
        n_steps: int,
    ) -> Tensor:
        """
        Compute multi-step variance forecast.

        For h-step ahead forecast, GARCH(1,1) gives:
            E[σ²_{t+h}|I_t] = σ² + (α+β)^{h-1} * (σ²_t - σ²)

        where σ² is the unconditional variance.

        Args:
            current_variance: Current conditional variance.
            n_steps: Number of steps ahead to forecast.

        Returns:
            Tensor of shape (batch, n_steps) with variance forecasts.
        """
        batch_size = current_variance.shape[0]
        device = current_variance.device

        forecasts = torch.zeros(batch_size, n_steps, device=device)
        uncond_var = self.unconditional_variance

        for h in range(n_steps):
            # h-step ahead forecast formula
            decay = self.persistence.pow(h)
            forecasts[:, h] = uncond_var + decay * (current_variance - uncond_var)

        return forecasts

    def garch_loss(
        self,
        returns: Tensor,
        variances: Tensor,
    ) -> Tensor:
        """
        Compute GARCH consistency loss.

        This loss measures how well the predicted variances satisfy
        the GARCH dynamics. Used as a physics-informed constraint.

        Args:
            returns: Actual returns of shape (batch, seq_len).
            variances: Predicted variances of shape (batch, seq_len).

        Returns:
            Scalar loss value.
        """
        shocks_sq = returns.pow(2)

        # Expected variance from GARCH recursion (for t > 0)
        expected_var = (
            self.omega
            + self.alpha * shocks_sq[:, :-1]
            + self.beta * variances[:, :-1]
        )

        # Actual variance at next time step
        actual_var = variances[:, 1:]

        # MSE between expected and actual
        return F.mse_loss(expected_var, actual_var)


class GARCHRegimeLayer(nn.Module):
    """
    Regime-switching GARCH layer.

    Allows different GARCH parameters for different market regimes
    (e.g., low volatility vs. high volatility periods). The regime
    is determined externally (e.g., by a regime classifier).

    This captures the empirical observation that volatility dynamics
    differ across market conditions:
    - Bull markets: Lower persistence, faster mean reversion
    - Bear markets: Higher persistence, volatility spikes
    """

    def __init__(
        self,
        n_regimes: int = 2,
        omega_init: Tuple[float, ...] = (0.0001, 0.0003),
        alpha_init: Tuple[float, ...] = (0.05, 0.15),
        beta_init: Tuple[float, ...] = (0.9, 0.8),
    ):
        """
        Initialize regime-switching GARCH.

        Args:
            n_regimes: Number of regimes.
            omega_init: Initial ω values for each regime.
            alpha_init: Initial α values for each regime.
            beta_init: Initial β values for each regime.
        """
        super().__init__()

        self.n_regimes = n_regimes

        # Create GARCH layer for each regime
        self.regime_garch = nn.ModuleList([
            GARCHLayer(
                omega=omega_init[i],
                alpha=alpha_init[i],
                beta=beta_init[i],
                learnable=True,
            )
            for i in range(n_regimes)
        ])

    def forward(
        self,
        returns: Tensor,
        regime_probs: Tensor,
        initial_variance: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute regime-weighted GARCH variances.

        Args:
            returns: Return series of shape (batch, seq_len).
            regime_probs: Regime probabilities of shape (batch, n_regimes).
            initial_variance: Optional initial variance.

        Returns:
            Weighted variance series of shape (batch, seq_len).
        """
        # Compute variance under each regime
        regime_variances = []
        for garch in self.regime_garch:
            var = garch(returns, initial_variance)
            regime_variances.append(var)

        # Stack: (batch, seq_len, n_regimes)
        variances = torch.stack(regime_variances, dim=-1)

        # Weight by regime probabilities
        # regime_probs: (batch, n_regimes) -> (batch, 1, n_regimes)
        weights = regime_probs.unsqueeze(1)

        # Weighted sum
        weighted_var = (variances * weights).sum(dim=-1)

        return weighted_var

    def get_regime_parameters(self) -> dict:
        """Get parameters for each regime."""
        params = {}
        for i, garch in enumerate(self.regime_garch):
            params[f'regime_{i}'] = {
                'omega': garch.omega.item(),
                'alpha': garch.alpha.item(),
                'beta': garch.beta.item(),
                'persistence': garch.persistence.item(),
            }
        return params
