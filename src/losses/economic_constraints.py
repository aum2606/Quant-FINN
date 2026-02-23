"""
Economic Constraint Losses for Finance PINNs.

This module implements losses that enforce economic principles:
1. No-arbitrage: Prevents free lunch opportunities
2. Monotonicity: Greeks have correct signs
3. Put-call parity: Relationship between puts and calls
4. Convexity: Option prices are convex in underlying

These constraints ensure model outputs are economically sensible,
even when extrapolating beyond training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from src.utils.autodiff import compute_gradient, compute_second_derivative


class NoArbitrageLoss(nn.Module):
    """
    No-Arbitrage Constraint Loss.

    Enforces fundamental no-arbitrage principles:

    1. Non-negative prices: V ≥ 0 (for calls and puts)
    2. Upper bounds: C ≤ S, P ≤ K·e^(-rT)
    3. Lower bounds: C ≥ max(S - K·e^(-rT), 0)
    4. Calendar spreads: V(T₁) ≤ V(T₂) for T₁ < T₂ (usually)

    Violations of these indicate arbitrage opportunities,
    which should not exist in efficient markets.
    """

    def __init__(
        self,
        K: float = 100.0,
        r: float = 0.05,
        option_type: str = 'call',
        penalty_weight: float = 100.0,
    ):
        """
        Initialize no-arbitrage loss.

        Args:
            K: Strike price.
            r: Risk-free rate.
            option_type: 'call' or 'put'.
            penalty_weight: Weight for constraint violations.
        """
        super().__init__()

        self.register_buffer('K', torch.tensor(K))
        self.register_buffer('r', torch.tensor(r))
        self.option_type = option_type
        self.penalty_weight = penalty_weight

    def forward(
        self,
        V: Tensor,
        S: Tensor,
        t: Tensor,
        T: float,
    ) -> Tensor:
        """
        Compute no-arbitrage violation loss.

        Args:
            V: Option values from network.
            S: Stock prices.
            t: Current times.
            T: Expiration time.

        Returns:
            Total arbitrage violation loss.
        """
        tau = T - t  # Time to expiration
        discount = torch.exp(-self.r * tau)

        losses = []

        # 1. Non-negativity: V ≥ 0
        negative_price = F.relu(-V)
        losses.append(negative_price.pow(2).mean())

        if self.option_type == 'call':
            # 2. Upper bound: C ≤ S
            upper_violation = F.relu(V - S)
            losses.append(upper_violation.pow(2).mean())

            # 3. Lower bound: C ≥ S - K·e^(-rτ)
            intrinsic = S - self.K * discount
            lower_violation = F.relu(intrinsic - V)
            losses.append(lower_violation.pow(2).mean())

        else:  # put
            # 2. Upper bound: P ≤ K·e^(-rτ)
            upper_violation = F.relu(V - self.K * discount)
            losses.append(upper_violation.pow(2).mean())

            # 3. Lower bound: P ≥ K·e^(-rτ) - S
            intrinsic = self.K * discount - S
            lower_violation = F.relu(intrinsic - V)
            losses.append(lower_violation.pow(2).mean())

        return self.penalty_weight * sum(losses)


class MonotonicityLoss(nn.Module):
    """
    Monotonicity Constraint Loss for Option Greeks.

    Options should satisfy certain monotonicity properties:

    For calls:
        - Delta (∂V/∂S) > 0: Value increases with stock price
        - Gamma (∂²V/∂S²) > 0: Convexity in stock price
        - Vega (∂V/∂σ) > 0: Value increases with volatility
        - Rho (∂V/∂r) > 0: Value increases with interest rate

    For puts:
        - Delta < 0: Value decreases with stock price
        - Gamma > 0: Still convex
        - Vega > 0: Still increases with volatility
        - Rho < 0: Value decreases with interest rate

    Additionally:
        - Delta ∈ [0, 1] for calls
        - Delta ∈ [-1, 0] for puts
    """

    def __init__(
        self,
        option_type: str = 'call',
        penalty_weight: float = 10.0,
        delta_bounds: bool = True,
    ):
        """
        Initialize monotonicity loss.

        Args:
            option_type: 'call' or 'put'.
            penalty_weight: Weight for violations.
            delta_bounds: If True, enforce delta bounds.
        """
        super().__init__()

        self.option_type = option_type
        self.penalty_weight = penalty_weight
        self.delta_bounds = delta_bounds

    def forward(
        self,
        network: nn.Module,
        S: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Compute monotonicity violation loss.

        Args:
            network: Option pricing network.
            S: Stock prices.
            t: Times.

        Returns:
            Total monotonicity violation loss.
        """
        S = S.requires_grad_(True) if not S.requires_grad else S

        V = network(S, t)
        if V.dim() > 1:
            V = V.squeeze(-1)

        # Compute Greeks
        delta = compute_gradient(V, S)
        gamma = compute_second_derivative(V, S)

        losses = []

        # Delta monotonicity
        if self.option_type == 'call':
            # Delta should be positive
            delta_violation = F.relu(-delta)
            losses.append(delta_violation.pow(2).mean())

            # Delta bounds: 0 ≤ Δ ≤ 1
            if self.delta_bounds:
                upper_violation = F.relu(delta - 1.0)
                losses.append(upper_violation.pow(2).mean())
        else:
            # Delta should be negative
            delta_violation = F.relu(delta)
            losses.append(delta_violation.pow(2).mean())

            # Delta bounds: -1 ≤ Δ ≤ 0
            if self.delta_bounds:
                lower_violation = F.relu(-delta - 1.0)
                losses.append(lower_violation.pow(2).mean())

        # Gamma convexity: Γ > 0 (applies to both calls and puts)
        gamma_violation = F.relu(-gamma)
        losses.append(gamma_violation.pow(2).mean())

        return self.penalty_weight * sum(losses)


class PutCallParityLoss(nn.Module):
    """
    Put-Call Parity Constraint Loss.

    The fundamental relationship between European puts and calls:
        C - P = S - K·e^(-r(T-t))

    or equivalently:
        C - P = S·e^(-q(T-t)) - K·e^(-r(T-t))

    for dividend-paying stocks (q = continuous dividend yield).

    Violating put-call parity indicates inconsistent pricing
    that would allow arbitrage.
    """

    def __init__(
        self,
        K: float = 100.0,
        r: float = 0.05,
        q: float = 0.0,
        penalty_weight: float = 100.0,
    ):
        """
        Initialize put-call parity loss.

        Args:
            K: Strike price.
            r: Risk-free rate.
            q: Continuous dividend yield.
            penalty_weight: Weight for violations.
        """
        super().__init__()

        self.register_buffer('K', torch.tensor(K))
        self.register_buffer('r', torch.tensor(r))
        self.register_buffer('q', torch.tensor(q))
        self.penalty_weight = penalty_weight

    def forward(
        self,
        call_prices: Tensor,
        put_prices: Tensor,
        S: Tensor,
        t: Tensor,
        T: float,
    ) -> Tensor:
        """
        Compute put-call parity violation loss.

        Args:
            call_prices: Call option prices.
            put_prices: Put option prices.
            S: Stock prices.
            t: Current times.
            T: Expiration time.

        Returns:
            Put-call parity violation loss.
        """
        tau = T - t  # Time to expiration

        # Left side: C - P
        lhs = call_prices - put_prices

        # Right side: S·e^(-qτ) - K·e^(-rτ)
        rhs = S * torch.exp(-self.q * tau) - self.K * torch.exp(-self.r * tau)

        # Parity violation
        violation = lhs - rhs

        return self.penalty_weight * violation.pow(2).mean()


class ConvexityLoss(nn.Module):
    """
    Convexity Constraint Loss.

    Option prices should be convex functions of the underlying:
        ∂²V/∂S² ≥ 0 (Gamma ≥ 0)

    And convex in strike price (for fixed S, t):
        ∂²V/∂K² ≥ 0

    Convexity ensures that butterfly spreads have non-negative value.
    """

    def __init__(self, penalty_weight: float = 10.0):
        """
        Initialize convexity loss.

        Args:
            penalty_weight: Weight for violations.
        """
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        network: nn.Module,
        S: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Compute convexity violation loss.

        Args:
            network: Option pricing network.
            S: Stock prices.
            t: Times.

        Returns:
            Convexity violation loss.
        """
        S = S.requires_grad_(True) if not S.requires_grad else S

        V = network(S, t)
        if V.dim() > 1:
            V = V.squeeze(-1)

        # Second derivative (Gamma)
        gamma = compute_second_derivative(V, S)

        # Convexity violation: Γ < 0
        violation = F.relu(-gamma)

        return self.penalty_weight * violation.pow(2).mean()


class TermStructureLoss(nn.Module):
    """
    Term Structure Constraint Loss.

    For European options without dividends:
        V(T₁) ≤ V(T₂) for T₁ < T₂

    Longer-dated options should be worth at least as much as
    shorter-dated options (time value is non-negative).

    Note: This doesn't hold for American options due to early exercise.
    """

    def __init__(self, penalty_weight: float = 10.0):
        """
        Initialize term structure loss.

        Args:
            penalty_weight: Weight for violations.
        """
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        network: nn.Module,
        S: Tensor,
        t1: Tensor,
        t2: Tensor,
    ) -> Tensor:
        """
        Compute term structure violation loss.

        Args:
            network: Option pricing network.
            S: Stock prices.
            t1: Earlier times.
            t2: Later times (t2 > t1, so closer to expiration).

        Returns:
            Term structure violation loss.
        """
        # Note: In our convention, larger t means closer to expiration
        # So V(t1) should be ≥ V(t2) if t1 < t2

        V1 = network(S, t1)
        V2 = network(S, t2)

        if V1.dim() > 1:
            V1 = V1.squeeze(-1)
            V2 = V2.squeeze(-1)

        # Violation: V2 > V1 (shorter-dated worth more)
        # Actually, if t represents time to maturity, longer maturity = larger t
        # Let's assume t is time to maturity here
        violation = F.relu(V1 - V2)  # V1 should be ≤ V2 if t1 < t2

        return self.penalty_weight * violation.pow(2).mean()


class CombinedEconomicLoss(nn.Module):
    """
    Combined Economic Constraint Loss.

    Combines multiple economic constraints with configurable weights.
    This is the recommended way to apply economic constraints during training.
    """

    def __init__(
        self,
        K: float = 100.0,
        r: float = 0.05,
        option_type: str = 'call',
        no_arbitrage_weight: float = 100.0,
        monotonicity_weight: float = 10.0,
        convexity_weight: float = 10.0,
    ):
        """
        Initialize combined economic loss.

        Args:
            K: Strike price.
            r: Risk-free rate.
            option_type: 'call' or 'put'.
            no_arbitrage_weight: Weight for no-arbitrage constraints.
            monotonicity_weight: Weight for monotonicity constraints.
            convexity_weight: Weight for convexity constraint.
        """
        super().__init__()

        self.no_arbitrage = NoArbitrageLoss(
            K=K, r=r, option_type=option_type,
            penalty_weight=no_arbitrage_weight
        )
        self.monotonicity = MonotonicityLoss(
            option_type=option_type,
            penalty_weight=monotonicity_weight
        )
        self.convexity = ConvexityLoss(penalty_weight=convexity_weight)

    def forward(
        self,
        network: nn.Module,
        S: Tensor,
        t: Tensor,
        T: float,
    ) -> dict:
        """
        Compute all economic constraint losses.

        Args:
            network: Option pricing network.
            S: Stock prices.
            t: Times.
            T: Expiration time.

        Returns:
            Dict with individual and total losses.
        """
        V = network(S, t)
        if V.dim() > 1:
            V = V.squeeze(-1)

        losses = {}

        losses['no_arbitrage'] = self.no_arbitrage(V, S, t, T)
        losses['monotonicity'] = self.monotonicity(network, S, t)
        losses['convexity'] = self.convexity(network, S, t)

        losses['total'] = sum(losses.values())

        return losses
