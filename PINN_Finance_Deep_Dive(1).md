# Physics-Informed Neural Networks (PINNs) for Finance
## A Comprehensive Deep Dive for Quantitative Trading Applications

---

## Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [PINN Architecture Deep Dive](#3-pinn-architecture-deep-dive)
4. [Financial PDEs to Embed](#4-financial-pdes-to-embed)
5. [Loss Function Engineering](#5-loss-function-engineering)
6. [Architecture Patterns for Finance](#6-architecture-patterns-for-finance)
7. [Training Strategies](#7-training-strategies)
8. [Design Decisions & Trade-offs](#8-design-decisions--trade-offs)
9. [Practical Considerations](#9-practical-considerations)
10. [Reference Architecture for Quant-FINN](#10-reference-architecture-for-quant-finn)

---

## 1. Foundational Concepts

### 1.1 What is a Physics-Informed Neural Network?

A PINN is a neural network that incorporates domain knowledge (expressed as differential equations, constraints, or physical laws) directly into its learning process. Instead of learning purely from data, the network learns to satisfy both:

1. **Data fidelity**: Match observed data points
2. **Physics/Domain constraints**: Satisfy known governing equations

```
Traditional NN:  Data → Network → Predictions
                        ↓
                   Minimize: ||predictions - data||²

PINN:            Data → Network → Predictions
                        ↓              ↓
                   Minimize: ||predictions - data||² + λ||PDE residual||²
                                                         ↑
                                            Domain knowledge embedded here
```

### 1.2 Why PINNs for Finance?

Financial markets are governed by well-established mathematical models:

| Domain | Governing Equations |
|--------|---------------------|
| Option Pricing | Black-Scholes PDE, Heston Model |
| Portfolio Theory | Mean-Variance Optimization |
| Risk Management | VaR equations, Greeks |
| Time Series | GARCH, Ornstein-Uhlenbeck |
| Market Microstructure | Order flow dynamics |

**The Problem with Pure ML in Finance:**
- Markets are non-stationary (distributions shift)
- Limited data for rare events (crashes, tail risks)
- Overfitting to noise is extremely easy
- No guarantee of economically sensible outputs

**How PINNs Help:**
- Regularize learning with economic theory
- Extrapolate better to unseen market conditions
- Produce outputs consistent with no-arbitrage principles
- Require less data by leveraging domain knowledge

### 1.3 The Core Insight

Consider the Black-Scholes equation for option pricing:

```
∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

A traditional approach: Solve this analytically or numerically, then use ML separately for prediction.

The PINN approach: Train a neural network where:
- Input: (S, t, σ, r, K) — stock price, time, volatility, rate, strike
- Output: V — option value
- Constraint: The output must satisfy the Black-Scholes PDE

The network learns the solution surface while being "informed" by the equation.

---

## 2. Mathematical Foundations

### 2.1 Partial Differential Equations (PDEs) Primer

A PDE relates a function to its partial derivatives. General form:

```
F(x, t, u, ∂u/∂x, ∂u/∂t, ∂²u/∂x², ...) = 0
```

Where:
- `u(x, t)` is the unknown function we want to find
- `x, t` are independent variables (space, time)
- `∂u/∂x` denotes partial derivative of u with respect to x

**Types of Conditions:**

1. **Initial Conditions (IC)**: Value of u at t=0
   ```
   u(x, 0) = g(x)  — "where does the system start?"
   ```

2. **Boundary Conditions (BC)**: Value of u at domain boundaries
   ```
   u(0, t) = h₁(t)   — "what happens at the left edge?"
   u(L, t) = h₂(t)   — "what happens at the right edge?"
   ```

3. **Terminal Conditions (TC)**: Value at final time (common in finance)
   ```
   u(x, T) = f(x)   — "what's the payoff at expiration?"
   ```

### 2.2 Automatic Differentiation: The Engine of PINNs

PINNs require computing derivatives of the neural network output with respect to its inputs. This is done via **automatic differentiation (autodiff)**.

**How it works:**

```python
# PyTorch example
import torch

# Network takes (S, t) as input, outputs V
S = torch.tensor([100.0], requires_grad=True)
t = torch.tensor([0.5], requires_grad=True)

# Forward pass
V = network(S, t)

# Compute first derivatives
dV_dS = torch.autograd.grad(V, S, create_graph=True)[0]
dV_dt = torch.autograd.grad(V, t, create_graph=True)[0]

# Compute second derivatives
d2V_dS2 = torch.autograd.grad(dV_dS, S, create_graph=True)[0]

# Now we can compute the PDE residual!
```

**Key Point:** `create_graph=True` allows gradients to flow through the derivative computation, enabling backpropagation through the PDE residual.

### 2.3 The Universal Approximation Connection

Neural networks are universal function approximators — they can approximate any continuous function to arbitrary precision given enough capacity.

PINNs leverage this by:
1. Parameterizing the solution u(x,t) as a neural network: u_θ(x,t)
2. Using autodiff to compute derivatives: ∂u_θ/∂x, ∂u_θ/∂t, etc.
3. Training θ to minimize the PDE residual

The network doesn't memorize solutions — it learns the **solution manifold**.

---

## 3. PINN Architecture Deep Dive

### 3.1 Basic PINN Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUTS                                   │
│         (x, t) or (S, t, σ, r, K) for finance                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT ENCODING LAYER                          │
│         (Optional: Fourier features, normalization)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HIDDEN LAYERS                                 │
│                                                                  │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│    │ Dense   │───▶│ Dense   │───▶│ Dense   │───▶│ Dense   │    │
│    │ + Act   │    │ + Act   │    │ + Act   │    │ + Act   │    │
│    └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│                                                                  │
│    Typical: 4-8 layers, 50-200 neurons each                     │
│    Activation: tanh (smooth derivatives) or swish               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
│              u_θ(x, t) — the solution approximation              │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   AUTODIFF    │ │   AUTODIFF    │ │   AUTODIFF    │
    │   ∂u/∂x       │ │   ∂u/∂t       │ │   ∂²u/∂x²     │
    └───────────────┘ └───────────────┘ └───────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PDE RESIDUAL COMPUTATION                      │
│                                                                  │
│    R = F(x, t, u, ∂u/∂x, ∂u/∂t, ∂²u/∂x², ...)                  │
│                                                                  │
│    For Black-Scholes:                                           │
│    R = ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LOSS COMPUTATION                            │
│                                                                  │
│    L = L_data + λ_pde·L_pde + λ_ic·L_ic + λ_bc·L_bc            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Activation Function Selection

**Critical for PINNs:** The activation function must have smooth, well-behaved derivatives since we compute ∂²u/∂x² and beyond.

| Activation | Formula | Pros | Cons |
|------------|---------|------|------|
| **tanh** | (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) | Smooth, bounded, standard choice | Vanishing gradients |
| **swish** | x · σ(x) | Smooth, non-monotonic, good gradients | Slightly slower |
| **GELU** | x · Φ(x) | Smooth, modern choice | Similar to swish |
| **sin** | sin(x) | Perfect for periodic solutions | Limited applicability |
| **softplus** | log(1 + eˣ) | Smooth, positive | One-sided |

**Avoid:** ReLU, LeakyReLU — their derivatives are discontinuous!

```python
# ReLU derivative
d/dx ReLU(x) = 0 if x < 0, 1 if x > 0, undefined at x = 0

# This causes problems when computing ∂²u/∂x²
```

### 3.3 Input Encoding Strategies

Raw inputs often lead to spectral bias — networks learn low-frequency components first and struggle with high-frequency details.

**Solution: Fourier Feature Encoding**

```python
def fourier_encoding(x, num_frequencies=10, scale=1.0):
    """
    Transform input x into Fourier features.
    
    x: input tensor of shape (batch, input_dim)
    Returns: encoded tensor of shape (batch, input_dim * 2 * num_frequencies)
    """
    frequencies = scale * torch.arange(1, num_frequencies + 1)
    
    # Compute sin and cos for each frequency
    x_proj = x.unsqueeze(-1) * frequencies  # (batch, input_dim, num_freq)
    
    sin_features = torch.sin(2 * np.pi * x_proj)
    cos_features = torch.cos(2 * np.pi * x_proj)
    
    # Flatten and concatenate
    encoded = torch.cat([sin_features, cos_features], dim=-1)
    return encoded.flatten(start_dim=1)
```

**Why this helps:**
- Maps low-dimensional inputs to high-dimensional space
- Different frequencies capture different scales of variation
- Empirically improves convergence for PDEs with complex solutions

### 3.4 Network Depth vs Width

**Empirical findings for PINNs:**

```
Shallow & Wide (2-3 layers, 500+ neurons):
├── Faster to train
├── Good for simple PDEs
└── May struggle with complex solutions

Deep & Narrow (6-10 layers, 50-100 neurons):
├── Better expressivity
├── Captures hierarchical features
├── Requires careful initialization
└── Benefits from residual connections

Recommended Starting Point:
├── 4-6 hidden layers
├── 100-200 neurons per layer
└── tanh or swish activation
```

### 3.5 Residual Connections for Deep PINNs

For deeper networks, residual connections help gradient flow:

```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.activation(x + residual)  # Skip connection
```

---

## 4. Financial PDEs to Embed

### 4.1 Black-Scholes PDE (Option Pricing)

The foundational equation for derivative pricing.

**The Equation:**
```
∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

**Variables:**
- V(S, t): Option value as function of stock price S and time t
- σ: Volatility (constant in basic model)
- r: Risk-free interest rate
- S: Underlying asset price
- t: Time (usually t=0 is now, t=T is expiration)

**Boundary/Terminal Conditions for European Call:**
```
Terminal:  V(S, T) = max(S - K, 0)           — payoff at expiration
Boundary:  V(0, t) = 0                        — worthless if S = 0
           V(S, t) → S - Ke^(-r(T-t)) as S → ∞  — deep in-the-money
```

**PINN Implementation:**

```python
def black_scholes_residual(network, S, t, sigma, r):
    """
    Compute the Black-Scholes PDE residual.
    
    Args:
        network: Neural network V_θ(S, t)
        S: Stock prices (batch,)
        t: Times (batch,)
        sigma: Volatility (scalar or batch)
        r: Risk-free rate (scalar or batch)
    
    Returns:
        residual: PDE residual (should be ≈ 0 if satisfied)
    """
    S = S.requires_grad_(True)
    t = t.requires_grad_(True)
    
    # Forward pass
    V = network(S, t)
    
    # First derivatives
    dV_dS = torch.autograd.grad(
        V, S, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]
    
    dV_dt = torch.autograd.grad(
        V, t, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]
    
    # Second derivative
    d2V_dS2 = torch.autograd.grad(
        dV_dS, S, grad_outputs=torch.ones_like(dV_dS),
        create_graph=True, retain_graph=True
    )[0]
    
    # Black-Scholes PDE residual
    residual = (
        dV_dt 
        + 0.5 * sigma**2 * S**2 * d2V_dS2 
        + r * S * dV_dS 
        - r * V
    )
    
    return residual
```

### 4.2 Heston Stochastic Volatility Model

More realistic than Black-Scholes — volatility itself is stochastic.

**The System:**
```
dS = μS dt + √v S dW₁
dv = κ(θ - v) dt + ξ√v dW₂

where: dW₁ · dW₂ = ρ dt  (correlated Brownian motions)
```

**Parameters:**
- κ: Mean reversion speed
- θ: Long-term variance level
- ξ: Volatility of volatility
- ρ: Correlation between price and volatility

**Heston PDE for Option Pricing:**
```
∂V/∂t + ½vS²(∂²V/∂S²) + ρξvS(∂²V/∂S∂v) + ½ξ²v(∂²V/∂v²)
      + rS(∂V/∂S) + [κ(θ-v) - λv](∂V/∂v) - rV = 0
```

This is a 2D PDE (in S and v), significantly more complex but still embeddable.

### 4.3 GARCH Volatility Model

For time series forecasting with volatility clustering.

**GARCH(1,1) Equation:**
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

where:
- σ²_t: Variance at time t
- ε_{t-1}: Previous period's shock (return - mean)
- ω, α, β: Parameters (ω > 0, α ≥ 0, β ≥ 0, α + β < 1)
```

**Embedding in PINN:**

```python
class GARCHInformedLayer(nn.Module):
    """
    Layer that enforces GARCH dynamics.
    """
    def __init__(self):
        super().__init__()
        # Learnable GARCH parameters (constrained to valid ranges)
        self.omega_raw = nn.Parameter(torch.tensor(0.0))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.beta_raw = nn.Parameter(torch.tensor(0.0))
    
    @property
    def omega(self):
        return torch.softplus(self.omega_raw) * 0.001  # Small positive
    
    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw) * 0.3  # [0, 0.3]
    
    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw) * 0.9  # [0, 0.9]
    
    def forward(self, epsilon_sq_prev, sigma_sq_prev):
        """
        Compute next period's variance following GARCH dynamics.
        """
        sigma_sq_next = (
            self.omega 
            + self.alpha * epsilon_sq_prev 
            + self.beta * sigma_sq_prev
        )
        return sigma_sq_next
    
    def persistence(self):
        """α + β should be < 1 for stationarity"""
        return self.alpha + self.beta
```

### 4.4 Mean-Variance Portfolio Optimization

Markowitz portfolio theory as a constraint.

**Optimization Problem:**
```
maximize:  μᵀw - (λ/2)wᵀΣw

subject to: Σᵢwᵢ = 1  (weights sum to 1)
            wᵢ ≥ 0     (optional: long-only)
```

**First-Order Condition (for unconstrained):**
```
μ - λΣw = ν·1  (where ν is Lagrange multiplier)

Optimal weights: w* = (1/λ)Σ⁻¹(μ - ν·1)
```

**PINN Approach:**

```python
def mean_variance_residual(network, returns_data, lambda_risk):
    """
    Network outputs portfolio weights w.
    Residual measures deviation from optimality conditions.
    """
    # Network predicts weights
    w = network(returns_data)  # (batch, n_assets)
    w = torch.softmax(w, dim=-1)  # Ensure sum to 1
    
    # Compute expected returns and covariance from data
    mu = returns_data.mean(dim=0)  # (n_assets,)
    Sigma = torch.cov(returns_data.T)  # (n_assets, n_assets)
    
    # First-order condition: μ - λΣw should be proportional to 1
    gradient = mu - lambda_risk * (Sigma @ w.T).T
    
    # Residual: gradient should be constant across assets
    residual = gradient - gradient.mean(dim=-1, keepdim=True)
    
    return residual.pow(2).mean()
```

### 4.5 Factor Models (Fama-French, APT)

Arbitrage Pricing Theory and factor models.

**General Factor Model:**
```
Rᵢ = αᵢ + Σⱼ βᵢⱼ·Fⱼ + εᵢ

where:
- Rᵢ: Return of asset i
- αᵢ: Alpha (excess return)
- βᵢⱼ: Exposure of asset i to factor j
- Fⱼ: Return of factor j
- εᵢ: Idiosyncratic error
```

**Fama-French 3-Factor:**
```
Rᵢ - Rᶠ = αᵢ + βᵐᵏᵗ(Rᵐ - Rᶠ) + βˢᵐᵇ·SMB + βʰᵐˡ·HML + εᵢ

where:
- Rᵐ - Rᶠ: Market excess return
- SMB: Small Minus Big (size factor)
- HML: High Minus Low (value factor)
```

**PINN Embedding:**

```python
class FactorInformedNetwork(nn.Module):
    """
    Network that respects factor model structure.
    """
    def __init__(self, n_assets, n_factors, hidden_dim):
        super().__init__()
        
        # Learn factor exposures (betas)
        self.beta_network = nn.Sequential(
            nn.Linear(n_assets, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_assets * n_factors)
        )
        
        # Learn alphas
        self.alpha_network = nn.Sequential(
            nn.Linear(n_assets, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_assets)
        )
        
        self.n_assets = n_assets
        self.n_factors = n_factors
    
    def forward(self, asset_features, factor_returns):
        """
        Predict asset returns using factor structure.
        
        asset_features: (batch, n_assets) - characteristics
        factor_returns: (batch, n_factors) - factor returns
        """
        # Get betas: (batch, n_assets, n_factors)
        betas = self.beta_network(asset_features)
        betas = betas.view(-1, self.n_assets, self.n_factors)
        
        # Get alphas: (batch, n_assets)
        alphas = self.alpha_network(asset_features)
        
        # Factor model: R = α + β·F
        factor_contribution = torch.einsum('baf,bf->ba', betas, factor_returns)
        predicted_returns = alphas + factor_contribution
        
        return predicted_returns, betas, alphas
    
    def factor_residual(self, predicted, actual, betas, factor_returns):
        """
        Residual: actual returns should follow factor structure.
        """
        factor_contribution = torch.einsum('baf,bf->ba', betas, factor_returns)
        
        # The unexplained part should have specific properties:
        # 1. Zero mean
        # 2. Uncorrelated with factors
        residual = actual - predicted
        
        # Orthogonality constraint: residuals uncorrelated with factors
        correlation_penalty = torch.einsum('ba,bf->af', residual, factor_returns)
        
        return correlation_penalty.pow(2).mean()
```

### 4.6 No-Arbitrage Constraints

Fundamental principle: No free lunch!

**Put-Call Parity:**
```
C - P = S - Ke^(-rT)

where:
- C: Call option price
- P: Put option price  
- S: Stock price
- K: Strike price
- r: Risk-free rate
- T: Time to expiration
```

**Implementation:**

```python
def no_arbitrage_loss(call_prices, put_prices, S, K, r, T):
    """
    Enforce put-call parity as a soft constraint.
    """
    parity_lhs = call_prices - put_prices
    parity_rhs = S - K * torch.exp(-r * T)
    
    violation = (parity_lhs - parity_rhs).pow(2)
    return violation.mean()
```

**Monotonicity Constraints:**
```
∂C/∂S > 0    — Call value increases with stock price
∂C/∂K < 0    — Call value decreases with strike
∂C/∂σ > 0    — Call value increases with volatility (Vega > 0)
∂C/∂T > 0    — Call value increases with time (for European)
```

---

## 5. Loss Function Engineering

### 5.1 Multi-Component Loss Structure

The total PINN loss combines multiple terms:

```
L_total = λ_data · L_data 
        + λ_pde · L_pde 
        + λ_ic · L_ic 
        + λ_bc · L_bc
        + λ_econ · L_economic
```

**Component Breakdown:**

```python
class PINNLoss:
    def __init__(self, lambda_data=1.0, lambda_pde=1.0, 
                 lambda_ic=1.0, lambda_bc=1.0, lambda_econ=1.0):
        self.weights = {
            'data': lambda_data,
            'pde': lambda_pde,
            'ic': lambda_ic,
            'bc': lambda_bc,
            'econ': lambda_econ
        }
    
    def compute(self, network, data_points, collocation_points, 
                ic_points, bc_points):
        """
        Compute total PINN loss.
        """
        losses = {}
        
        # 1. Data Loss: Match observed market data
        if data_points is not None:
            pred = network(data_points['inputs'])
            losses['data'] = F.mse_loss(pred, data_points['targets'])
        
        # 2. PDE Loss: Satisfy governing equation at collocation points
        residual = self.compute_pde_residual(network, collocation_points)
        losses['pde'] = residual.pow(2).mean()
        
        # 3. Initial Condition Loss
        pred_ic = network(ic_points['inputs'])
        losses['ic'] = F.mse_loss(pred_ic, ic_points['targets'])
        
        # 4. Boundary Condition Loss
        pred_bc = network(bc_points['inputs'])
        losses['bc'] = F.mse_loss(pred_bc, bc_points['targets'])
        
        # 5. Economic Constraints Loss
        losses['econ'] = self.compute_economic_constraints(network)
        
        # Weighted sum
        total = sum(self.weights[k] * v for k, v in losses.items())
        
        return total, losses
```

### 5.2 Collocation Points Sampling

**Collocation points** are locations in the input domain where we enforce the PDE. They don't need labels — we just require the PDE residual to be zero.

**Sampling Strategies:**

```python
def sample_collocation_points(n_points, domain_bounds, strategy='uniform'):
    """
    Sample points in the domain for PDE enforcement.
    
    domain_bounds: dict with 'S': (S_min, S_max), 't': (t_min, t_max)
    """
    if strategy == 'uniform':
        # Random uniform sampling
        S = torch.rand(n_points) * (domain_bounds['S'][1] - domain_bounds['S'][0])
        S = S + domain_bounds['S'][0]
        t = torch.rand(n_points) * (domain_bounds['t'][1] - domain_bounds['t'][0])
        t = t + domain_bounds['t'][0]
    
    elif strategy == 'latin_hypercube':
        # Latin Hypercube Sampling — better space coverage
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=2)
        samples = sampler.random(n=n_points)
        
        S = samples[:, 0] * (domain_bounds['S'][1] - domain_bounds['S'][0])
        S = S + domain_bounds['S'][0]
        t = samples[:, 1] * (domain_bounds['t'][1] - domain_bounds['t'][0])
        t = t + domain_bounds['t'][0]
    
    elif strategy == 'importance':
        # Sample more points near boundaries and strike price
        # (where the solution has more structure)
        # ... custom implementation
        pass
    
    return torch.tensor(S), torch.tensor(t)
```

**Adaptive Sampling:** Sample more points where residual is high.

```python
def adaptive_collocation_sampling(network, current_points, n_new_points):
    """
    Add new collocation points where PDE residual is largest.
    """
    with torch.no_grad():
        residuals = compute_pde_residual(network, current_points)
        residual_magnitude = residuals.abs()
    
    # Sample new points with probability proportional to residual
    probs = residual_magnitude / residual_magnitude.sum()
    indices = torch.multinomial(probs, n_new_points, replacement=True)
    
    # Perturb selected points slightly
    selected = current_points[indices]
    noise = torch.randn_like(selected) * 0.01
    new_points = selected + noise
    
    return torch.cat([current_points, new_points], dim=0)
```

### 5.3 Loss Weighting Strategies

Balancing loss components is crucial — poor weighting leads to one term dominating.

**Strategy 1: Manual Tuning**
```python
# Start with equal weights, adjust based on loss magnitudes
lambda_data = 1.0
lambda_pde = 0.1  # Often needs to be smaller (PDE loss can be large)
lambda_bc = 10.0  # Boundary conditions are critical
```

**Strategy 2: Gradient-Based Normalization**
```python
def normalized_weights(losses, gradients):
    """
    Weight inversely proportional to gradient magnitude.
    """
    grad_norms = {k: g.norm() for k, g in gradients.items()}
    mean_norm = sum(grad_norms.values()) / len(grad_norms)
    
    weights = {k: mean_norm / (gn + 1e-8) for k, gn in grad_norms.items()}
    return weights
```

**Strategy 3: Learning Rate Annealing**
```python
class AnnealingScheduler:
    """
    Increase PDE weight over time as network learns data first.
    """
    def __init__(self, initial_pde_weight=0.01, final_pde_weight=1.0, 
                 warmup_epochs=100):
        self.initial = initial_pde_weight
        self.final = final_pde_weight
        self.warmup = warmup_epochs
    
    def get_weight(self, epoch):
        if epoch < self.warmup:
            # Linear warmup
            progress = epoch / self.warmup
            return self.initial + (self.final - self.initial) * progress
        return self.final
```

### 5.4 Handling Stiff PDEs

Some financial PDEs are "stiff" — they have widely different time scales that cause training difficulties.

**Solutions:**

1. **Log-Transform Variables**
```python
# Instead of V(S, t), use V(log(S), t)
log_S = torch.log(S)
V = network(log_S, t)
```

2. **Causal Training** (enforce causality in time)
```python
def causal_loss(residuals, time_points, epsilon=1.0):
    """
    Weight residuals by causal factor.
    Earlier times should satisfy PDE before later times.
    """
    sorted_indices = torch.argsort(time_points)
    sorted_residuals = residuals[sorted_indices]
    sorted_times = time_points[sorted_indices]
    
    # Cumulative squared residuals
    cumsum = torch.cumsum(sorted_residuals.pow(2), dim=0)
    weights = torch.exp(-epsilon * cumsum.detach())
    
    return (weights * sorted_residuals.pow(2)).mean()
```

---

## 6. Architecture Patterns for Finance

### 6.1 Pattern: Separate Networks for Each Output

When predicting multiple related quantities (price, delta, vega), use separate network heads:

```
                    Shared Encoder
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │ Price   │   │ Delta   │   │ Vega    │
      │ Head    │   │ Head    │   │ Head    │
      └─────────┘   └─────────┘   └─────────┘
           │             │             │
           ▼             ▼             ▼
           V            ∂V/∂S        ∂V/∂σ
```

**Advantage:** Can enforce relationships between outputs (delta IS ∂V/∂S).

```python
class MultiOutputPINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        
        # Shared encoder
        self.encoder = self._make_encoder(input_dim, hidden_dim, n_layers)
        
        # Separate heads
        self.price_head = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.gamma_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, S, t, sigma, r, K):
        x = torch.stack([S, t, sigma, r, K], dim=-1)
        features = self.encoder(x)
        
        V = self.price_head(features)
        delta_pred = self.delta_head(features)
        gamma_pred = self.gamma_head(features)
        
        return V, delta_pred, gamma_pred
    
    def consistency_loss(self, S, t, sigma, r, K):
        """
        Ensure delta_pred = ∂V/∂S from autodiff.
        """
        S = S.requires_grad_(True)
        V, delta_pred, gamma_pred = self.forward(S, t, sigma, r, K)
        
        # Autodiff delta
        delta_autodiff = torch.autograd.grad(
            V, S, grad_outputs=torch.ones_like(V),
            create_graph=True
        )[0]
        
        # Loss: predicted delta should match autodiff delta
        return F.mse_loss(delta_pred.squeeze(), delta_autodiff)
```

### 6.2 Pattern: Encoder-Decoder for Time Series

For sequential financial data (returns, prices over time):

```
Historical Data                          Future Predictions
[r_{t-N}, ..., r_{t-1}, r_t]  ───▶  Encoder  ───▶  Latent  ───▶  Decoder  ───▶  [r_{t+1}, ..., r_{t+H}]
                                                     │
                                                     │
                                              PINN Constraints
                                        (GARCH dynamics, etc.)
```

```python
class TimeSeriesPINN(nn.Module):
    def __init__(self, seq_len, hidden_dim, pred_horizon):
        super().__init__()
        
        # Encoder: LSTM to capture temporal patterns
        self.encoder = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_dim, 
            num_layers=2,
            batch_first=True
        )
        
        # Latent processing with PINN structure
        self.latent_pinn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, pred_horizon)
        
        # GARCH layer for volatility
        self.garch = GARCHInformedLayer()
    
    def forward(self, x):
        # x: (batch, seq_len, 1)
        encoded, (h_n, c_n) = self.encoder(x)
        latent = h_n[-1]  # (batch, hidden_dim)
        
        processed = self.latent_pinn(latent)
        predictions = self.decoder(processed)
        
        return predictions
    
    def garch_consistency_loss(self, returns):
        """
        Ensure predicted volatility follows GARCH dynamics.
        """
        # Compute rolling volatility from returns
        returns_sq = returns.pow(2)
        
        losses = []
        sigma_sq = returns_sq[:, 0]  # Initial variance
        
        for t in range(1, returns.shape[1]):
            sigma_sq_pred = self.garch(returns_sq[:, t-1], sigma_sq)
            sigma_sq_actual = returns_sq[:, t]  # Proxy for variance
            
            losses.append(F.mse_loss(sigma_sq_pred, sigma_sq_actual))
            sigma_sq = sigma_sq_pred
        
        return torch.stack(losses).mean()
```

### 6.3 Pattern: Hierarchical PINN for Multi-Scale Problems

Financial markets operate at multiple time scales (seconds to years):

```
                    ┌─────────────────────────────┐
                    │     Macro Level PINN        │
                    │  (Economic cycles, trends)  │
                    │  PDE: Long-term dynamics    │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │     Meso Level PINN         │
                    │  (Daily/weekly patterns)    │
                    │  PDE: Volatility dynamics   │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │     Micro Level PINN        │
                    │  (Intraday, tick-by-tick)   │
                    │  PDE: Market microstructure │
                    └─────────────────────────────┘
```

### 6.4 Pattern: Conditional PINN for Regime-Dependent Dynamics

Markets behave differently in different regimes (bull/bear, high/low volatility):

```python
class RegimeConditionalPINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_regimes):
        super().__init__()
        
        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Separate PINN for each regime
        self.regime_pinns = nn.ModuleList([
            self._make_pinn(input_dim, hidden_dim) 
            for _ in range(n_regimes)
        ])
        
        # Regime-specific PDE parameters
        self.regime_params = nn.ParameterList([
            nn.ParameterDict({
                'sigma': nn.Parameter(torch.tensor(0.2 + 0.1 * i)),
                'kappa': nn.Parameter(torch.tensor(1.0 + 0.5 * i))
            })
            for i in range(n_regimes)
        ])
    
    def forward(self, x):
        # Get regime probabilities
        regime_probs = self.regime_classifier(x)  # (batch, n_regimes)
        
        # Get predictions from each regime-specific PINN
        regime_outputs = torch.stack([
            pinn(x) for pinn in self.regime_pinns
        ], dim=-1)  # (batch, output_dim, n_regimes)
        
        # Mixture of experts
        output = (regime_outputs * regime_probs.unsqueeze(1)).sum(dim=-1)
        
        return output, regime_probs
    
    def pde_loss(self, x):
        """
        PDE loss weighted by regime probability.
        """
        regime_probs = self.regime_classifier(x)
        
        total_loss = 0
        for i, (pinn, params) in enumerate(zip(self.regime_pinns, self.regime_params)):
            residual = self.compute_residual(pinn, x, params)
            regime_weight = regime_probs[:, i]
            total_loss += (regime_weight * residual.pow(2)).mean()
        
        return total_loss
```

---

## 7. Training Strategies

### 7.1 Two-Stage Training

**Stage 1: Learn from Data**
```python
# First, train primarily on data to get reasonable initial solution
for epoch in range(pretrain_epochs):
    loss = data_loss(network, data)
    loss.backward()
    optimizer.step()
```

**Stage 2: Enforce PDE**
```python
# Then, add PDE constraints
for epoch in range(main_epochs):
    loss = data_loss + lambda_pde * pde_loss
    loss.backward()
    optimizer.step()
```

### 7.2 Curriculum Learning

Start with easier problems, gradually increase difficulty:

```python
class CurriculumTrainer:
    def __init__(self, network):
        self.network = network
        self.difficulty = 0.0  # 0 = easiest, 1 = full problem
    
    def train_step(self, batch):
        # Gradually increase difficulty
        self.difficulty = min(1.0, self.difficulty + 0.001)
        
        # Easy: Only interior points, far from boundaries
        # Hard: Include boundary points, singularities
        
        if self.difficulty < 0.5:
            # Use only well-behaved collocation points
            collocation = self.sample_easy_points()
        else:
            # Include challenging regions
            easy_frac = 2 * (1 - self.difficulty)  # 1 -> 0
            collocation = self.sample_mixed_points(easy_frac)
        
        loss = self.compute_loss(collocation)
        return loss
```

### 7.3 Optimizers for PINNs

**Adam** is a good default, but consider:

**L-BFGS** for final refinement (quasi-Newton method):
```python
# After Adam training, fine-tune with L-BFGS
optimizer = torch.optim.LBFGS(
    network.parameters(),
    lr=1.0,
    max_iter=500,
    history_size=50
)

def closure():
    optimizer.zero_grad()
    loss = compute_total_loss()
    loss.backward()
    return loss

optimizer.step(closure)
```

**Learning Rate Scheduling:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=100,
    min_lr=1e-6
)
```

### 7.4 Gradient Balancing

When gradients from different loss terms conflict:

```python
def gradient_surgery(gradients_list):
    """
    Project conflicting gradients onto each other.
    """
    # Stack gradients
    grads = torch.stack(gradients_list)  # (n_tasks, param_dim)
    
    # Compute pairwise dot products
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            dot = (grads[i] * grads[j]).sum()
            if dot < 0:  # Conflicting
                # Project grad_i onto orthogonal complement of grad_j
                grads[i] = grads[i] - (dot / (grads[j].norm()**2 + 1e-8)) * grads[j]
    
    return grads.mean(dim=0)  # Averaged gradient
```

---

## 8. Design Decisions & Trade-offs

### 8.1 Soft vs Hard Constraints

| Approach | Implementation | Pros | Cons |
|----------|----------------|------|------|
| **Soft** | Add to loss function | Flexible, easy to implement | May violate constraints |
| **Hard** | Architectural constraint | Guarantees satisfaction | Less flexible, harder to design |

**Example: Ensuring positive prices**

```python
# Soft constraint
positive_penalty = F.relu(-predicted_price).pow(2).mean()
loss = prediction_loss + 100 * positive_penalty

# Hard constraint
predicted_price = F.softplus(network_output)  # Always positive
```

### 8.2 Data-Driven vs Physics-Driven

```
More Data-Driven (λ_data >> λ_pde):
├── Better fit to observed data
├── Risk of overfitting
├── May produce physically implausible results
└── Use when: Lots of high-quality data, unknown physics

More Physics-Driven (λ_pde >> λ_data):
├── Smoother, more regular solutions
├── Better extrapolation
├── May not fit data well if physics is incomplete
└── Use when: Limited data, well-understood dynamics
```

### 8.3 Network Size Considerations

```
Small Network (< 10K params):
├── Fast training
├── Less prone to overfitting
├── May underfit complex solutions
└── Good for: Simple PDEs, prototype testing

Large Network (> 100K params):
├── Can capture complex solutions
├── Slower training
├── Needs more collocation points
└── Good for: Multi-dimensional problems, complex dynamics
```

### 8.4 When PINNs May Not Help

Be cautious if:
1. **Physics is unknown or misspecified** — Wrong equations hurt more than help
2. **Abundant high-quality data** — Pure ML may work better
3. **Discontinuous solutions** — PDEs assume smoothness
4. **Very high dimensions** — Curse of dimensionality still applies

---

## 9. Practical Considerations

### 9.1 Normalization

**Always normalize inputs and outputs!**

```python
class Normalizer:
    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
    
    def fit(self, inputs, outputs):
        self.input_mean = inputs.mean(dim=0)
        self.input_std = inputs.std(dim=0) + 1e-8
        self.output_mean = outputs.mean()
        self.output_std = outputs.std() + 1e-8
    
    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std
    
    def normalize_output(self, y):
        return (y - self.output_mean) / self.output_std
    
    def denormalize_output(self, y_norm):
        return y_norm * self.output_std + self.output_mean
```

**Important:** When computing PDE residuals with normalized variables, account for the chain rule!

```python
# If V_norm = (V - V_mean) / V_std and S_norm = (S - S_mean) / S_std
# Then: dV/dS = (V_std / S_std) * dV_norm/dS_norm
```

### 9.2 Dealing with Noisy Financial Data

Financial data is notoriously noisy. Options:

1. **Robust loss functions**
```python
def huber_loss(pred, target, delta=1.0):
    error = (pred - target).abs()
    return torch.where(
        error < delta,
        0.5 * error.pow(2),
        delta * (error - 0.5 * delta)
    ).mean()
```

2. **Data augmentation with noise**
```python
def add_noise(data, noise_level=0.01):
    noise = torch.randn_like(data) * noise_level
    return data + noise
```

3. **Ensemble of PINNs**
```python
class EnsemblePINN:
    def __init__(self, n_models):
        self.models = [PINN() for _ in range(n_models)]
    
    def predict(self, x):
        predictions = [m(x) for m in self.models]
        mean = torch.stack(predictions).mean(dim=0)
        std = torch.stack(predictions).std(dim=0)
        return mean, std  # Uncertainty estimate!
```

### 9.3 Validation Strategy

1. **Hold out time periods** (not random points!)
2. **Validate on different market regimes**
3. **Check PDE residual on unseen points**
4. **Compare Greeks to finite differences**

```python
def validate_greeks(network, S, t, dS=0.01):
    """
    Validate that network derivatives match finite differences.
    """
    V = network(S, t)
    V_up = network(S + dS, t)
    V_down = network(S - dS, t)
    
    # Finite difference delta and gamma
    delta_fd = (V_up - V_down) / (2 * dS)
    gamma_fd = (V_up - 2 * V + V_down) / (dS ** 2)
    
    # Autodiff delta and gamma
    S_ad = S.requires_grad_(True)
    V_ad = network(S_ad, t)
    delta_ad = torch.autograd.grad(V_ad, S_ad, create_graph=True)[0]
    gamma_ad = torch.autograd.grad(delta_ad, S_ad)[0]
    
    # Compare
    delta_error = (delta_fd - delta_ad).abs().mean()
    gamma_error = (gamma_fd - gamma_ad).abs().mean()
    
    return delta_error, gamma_error
```

### 9.4 Debugging PINNs

Common issues and solutions:

| Issue | Symptom | Solution |
|-------|---------|----------|
| PDE loss doesn't decrease | Residual stays high | Check autodiff, verify equation |
| Training unstable | Loss oscillates | Reduce learning rate, normalize |
| Poor extrapolation | Bad on OOD points | Increase PDE weight, add collocation points |
| BC/IC not satisfied | Boundary errors | Increase BC/IC weights, hard enforce |
| Slow convergence | 10K+ epochs needed | Fourier features, better initialization |

**Diagnostic plotting:**

```python
def plot_diagnostics(network, S_range, t_range):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Solution surface
    S_grid, t_grid = torch.meshgrid(S_range, t_range)
    V = network(S_grid.flatten(), t_grid.flatten()).reshape(S_grid.shape)
    axes[0, 0].contourf(S_grid, t_grid, V.detach())
    axes[0, 0].set_title('Solution V(S, t)')
    
    # 2. PDE residual
    residual = compute_pde_residual(network, S_grid.flatten(), t_grid.flatten())
    residual = residual.reshape(S_grid.shape)
    axes[0, 1].contourf(S_grid, t_grid, residual.detach().abs())
    axes[0, 1].set_title('|PDE Residual|')
    
    # 3. Delta
    delta = compute_delta(network, S_range, t_range[0])
    axes[1, 0].plot(S_range.detach(), delta.detach())
    axes[1, 0].set_title('Delta at t=0')
    
    # 4. Terminal condition check
    V_terminal = network(S_range, torch.full_like(S_range, t_range[-1]))
    payoff = torch.relu(S_range - K)
    axes[1, 1].plot(S_range.detach(), V_terminal.detach(), label='Network')
    axes[1, 1].plot(S_range.detach(), payoff.detach(), '--', label='Payoff')
    axes[1, 1].set_title('Terminal Condition')
    axes[1, 1].legend()
    
    plt.tight_layout()
    return fig
```

---

## 10. Reference Architecture for Quant-FINN

Based on all the above, here's a comprehensive architecture for your quantitative trading PINN:

### 10.1 Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Price Data  │  │ Technical   │  │ Fundamental │  │ Sentiment   │        │
│  │ (OHLCV)     │  │ Indicators  │  │ Factors     │  │ (NLP)       │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│        │                │                │                │                 │
└────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE ENCODING                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Temporal    │  │ Fourier     │  │ Factor      │  │ Sentiment   │        │
│  │ Encoder     │  │ Features    │  │ Embedding   │  │ Encoder     │        │
│  │ (LSTM)      │  │ Encoding    │  │             │  │ (FinBERT)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│        │                │                │                │                 │
│        └────────────────┴────────────────┴────────────────┘                 │
│                                   │                                          │
│                        ┌──────────┴──────────┐                              │
│                        │   Feature Fusion    │                              │
│                        │   (Attention/Concat)│                              │
│                        └──────────┬──────────┘                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHYSICS-INFORMED CORE                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     REGIME DETECTION                                 │   │
│  │         (Hidden Markov Model / Learned Clustering)                   │   │
│  │     ┌──────────┐    ┌──────────┐    ┌──────────┐                    │   │
│  │     │ Bull     │    │ Bear     │    │ Sideways │                    │   │
│  │     │ Regime   │    │ Regime   │    │ Regime   │                    │   │
│  │     └────┬─────┘    └────┬─────┘    └────┬─────┘                    │   │
│  └──────────┼───────────────┼───────────────┼──────────────────────────┘   │
│             │               │               │                               │
│             ▼               ▼               ▼                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              REGIME-SPECIFIC EQUATION LAYERS                         │   │
│  │                                                                      │   │
│  │   ┌────────────────────────────────────────────────────────────┐   │   │
│  │   │  GARCH Volatility Layer                                    │   │   │
│  │   │  σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}                       │   │   │
│  │   │  [Learnable: ω, α, β per regime]                           │   │   │
│  │   └────────────────────────────────────────────────────────────┘   │   │
│  │                            │                                        │   │
│  │   ┌────────────────────────────────────────────────────────────┐   │   │
│  │   │  Factor Model Layer                                        │   │   │
│  │   │  R_i = α_i + Σ_j β_ij · F_j + ε_i                         │   │   │
│  │   │  [Learnable: betas, constrained alphas]                    │   │   │
│  │   └────────────────────────────────────────────────────────────┘   │   │
│  │                            │                                        │   │
│  │   ┌────────────────────────────────────────────────────────────┐   │   │
│  │   │  Risk Parity / Mean-Variance Layer                         │   │   │
│  │   │  Enforces: μᵀw - (λ/2)wᵀΣw optimality                     │   │   │
│  │   └────────────────────────────────────────────────────────────┘   │   │
│  │                            │                                        │   │
│  │   ┌────────────────────────────────────────────────────────────┐   │   │
│  │   │  No-Arbitrage Constraints                                  │   │   │
│  │   │  - Put-Call Parity                                         │   │   │
│  │   │  - Monotonicity of Greeks                                  │   │   │
│  │   └────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT HEADS                                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐  │
│  │ Price/Return  │  │ Volatility    │  │ Position      │  │ Confidence  │  │
│  │ Prediction    │  │ Forecast      │  │ Sizing        │  │ Score       │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  └─────────────┘  │
│        │                  │                  │                  │           │
│        ▼                  ▼                  ▼                  ▼           │
│   Direction &        σ_t+1, σ_t+2       Optimal w_i        Epistemic      │
│   Magnitude          ...                                   Uncertainty     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LOSS FUNCTION                                          │
│                                                                              │
│   L_total = λ₁·L_prediction  (MSE on returns)                               │
│           + λ₂·L_GARCH       (GARCH dynamics residual)                      │
│           + λ₃·L_factor      (Factor model structure)                       │
│           + λ₄·L_portfolio   (Mean-variance optimality)                     │
│           + λ₅·L_arbitrage   (No-arbitrage constraints)                     │
│           + λ₆·L_regime      (Regime consistency)                           │
│           + λ₇·L_regularization                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Implementation Skeleton

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantFINN(nn.Module):
    """
    Quantitative Finance-Informed Neural Network
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # === Feature Encoders ===
        self.price_encoder = TemporalEncoder(
            input_dim=config.price_features,
            hidden_dim=config.hidden_dim,
            n_layers=config.encoder_layers
        )
        
        self.sentiment_encoder = SentimentEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim
        )
        
        self.fourier_encoder = FourierFeatureEncoder(
            input_dim=config.technical_features,
            num_frequencies=config.n_frequencies
        )
        
        # === Feature Fusion ===
        total_encoded_dim = config.hidden_dim * 3  # From 3 encoders
        self.fusion = nn.MultiheadAttention(
            embed_dim=total_encoded_dim,
            num_heads=config.n_attention_heads
        )
        
        # === Regime Detection ===
        self.regime_classifier = nn.Sequential(
            nn.Linear(total_encoded_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.n_regimes),
            nn.Softmax(dim=-1)
        )
        
        # === Physics-Informed Layers ===
        self.garch_layers = nn.ModuleList([
            GARCHLayer() for _ in range(config.n_regimes)
        ])
        
        self.factor_model = FactorModelLayer(
            n_assets=config.n_assets,
            n_factors=config.n_factors,
            hidden_dim=config.hidden_dim
        )
        
        self.portfolio_optimizer = MeanVarianceLayer(
            n_assets=config.n_assets,
            risk_aversion=config.risk_aversion
        )
        
        # === Output Heads ===
        self.return_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, config.n_assets)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, config.n_assets),
            nn.Softplus()  # Ensure positive volatility
        )
        
        self.position_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, config.n_assets),
            nn.Tanh()  # Positions in [-1, 1]
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()  # Confidence in [0, 1]
        )
    
    def forward(self, price_data, technical_data, fundamental_data, sentiment_data):
        """
        Forward pass through the network.
        
        Returns:
            predictions: dict with 'returns', 'volatility', 'positions', 'confidence'
            regime_probs: probability distribution over regimes
            intermediate: dict with intermediate values for loss computation
        """
        # Encode features
        price_features = self.price_encoder(price_data)
        technical_features = self.fourier_encoder(technical_data)
        sentiment_features = self.sentiment_encoder(sentiment_data)
        
        # Fuse features
        combined = torch.cat([price_features, technical_features, sentiment_features], dim=-1)
        fused, _ = self.fusion(combined, combined, combined)
        
        # Detect regime
        regime_probs = self.regime_classifier(fused)
        
        # Apply regime-specific GARCH
        volatility_components = []
        for i, garch in enumerate(self.garch_layers):
            vol = garch(price_data)  # Simplified
            volatility_components.append(vol * regime_probs[:, i:i+1])
        volatility_forecast = sum(volatility_components)
        
        # Factor model returns
        factor_returns, betas, alphas = self.factor_model(
            fundamental_data, 
            self.extract_factors(price_data)
        )
        
        # Generate outputs
        hidden = fused  # Or further process
        
        predicted_returns = self.return_head(hidden)
        predicted_volatility = self.volatility_head(hidden)
        predicted_positions = self.position_head(hidden)
        confidence = self.confidence_head(hidden)
        
        # Optimal positions from mean-variance
        optimal_positions = self.portfolio_optimizer(
            predicted_returns, 
            predicted_volatility
        )
        
        predictions = {
            'returns': predicted_returns,
            'volatility': predicted_volatility,
            'positions': predicted_positions,
            'optimal_positions': optimal_positions,
            'confidence': confidence
        }
        
        intermediate = {
            'factor_returns': factor_returns,
            'betas': betas,
            'alphas': alphas,
            'volatility_forecast': volatility_forecast
        }
        
        return predictions, regime_probs, intermediate
    
    def compute_total_loss(self, predictions, intermediate, regime_probs, 
                           actual_returns, actual_volatility):
        """
        Compute the full PINN loss.
        """
        losses = {}
        
        # 1. Prediction loss
        losses['returns'] = F.mse_loss(predictions['returns'], actual_returns)
        losses['volatility'] = F.mse_loss(predictions['volatility'], actual_volatility)
        
        # 2. GARCH consistency
        losses['garch'] = self.garch_consistency_loss(
            intermediate['volatility_forecast'],
            actual_volatility
        )
        
        # 3. Factor model structure
        losses['factor'] = self.factor_structure_loss(
            intermediate['factor_returns'],
            actual_returns,
            intermediate['betas']
        )
        
        # 4. Portfolio optimality
        losses['portfolio'] = self.portfolio_optimality_loss(
            predictions['positions'],
            predictions['optimal_positions']
        )
        
        # 5. Regime consistency
        losses['regime'] = self.regime_consistency_loss(regime_probs)
        
        # Weighted sum
        weights = self.config.loss_weights
        total = sum(weights[k] * v for k, v in losses.items())
        
        return total, losses


# Supporting layer classes

class GARCHLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(0.0001))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.8))
    
    def forward(self, returns):
        # Ensure valid parameters
        omega = F.softplus(self.omega) * 0.0001
        alpha = torch.sigmoid(self.alpha) * 0.3
        beta = torch.sigmoid(self.beta) * 0.9
        
        batch_size, seq_len = returns.shape
        
        # Initialize variance
        variance = returns.var(dim=1, keepdim=True)
        variances = [variance]
        
        for t in range(1, seq_len):
            shock_sq = returns[:, t-1:t] ** 2
            variance = omega + alpha * shock_sq + beta * variance
            variances.append(variance)
        
        return torch.cat(variances, dim=1)


class FactorModelLayer(nn.Module):
    def __init__(self, n_assets, n_factors, hidden_dim):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        
        self.beta_net = nn.Sequential(
            nn.Linear(n_assets, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_assets * n_factors)
        )
        
        self.alpha_net = nn.Sequential(
            nn.Linear(n_assets, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_assets)
        )
    
    def forward(self, characteristics, factor_returns):
        betas = self.beta_net(characteristics).view(-1, self.n_assets, self.n_factors)
        alphas = self.alpha_net(characteristics)
        
        # R = α + β·F
        factor_contribution = torch.einsum('baf,bf->ba', betas, factor_returns)
        predicted_returns = alphas + factor_contribution
        
        return predicted_returns, betas, alphas


class MeanVarianceLayer(nn.Module):
    def __init__(self, n_assets, risk_aversion=1.0):
        super().__init__()
        self.risk_aversion = risk_aversion
    
    def forward(self, expected_returns, volatility):
        # Simplified mean-variance: w_i ∝ μ_i / σ_i²
        risk_adjusted = expected_returns / (volatility ** 2 + 1e-6)
        weights = F.softmax(risk_adjusted / self.risk_aversion, dim=-1)
        return weights
```

### 10.3 Training Loop Skeleton

```python
def train_quant_finn(model, train_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.patience, factor=0.5
    )
    
    # Loss weight annealing
    pde_weight_scheduler = AnnealingScheduler(
        initial=0.01, final=1.0, warmup_epochs=config.warmup_epochs
    )
    
    for epoch in range(config.n_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        
        # Update PDE weight
        config.loss_weights['garch'] = pde_weight_scheduler.get_weight(epoch)
        config.loss_weights['factor'] = pde_weight_scheduler.get_weight(epoch)
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Unpack batch
            price_data = batch['prices']
            technical_data = batch['technical']
            fundamental_data = batch['fundamental']
            sentiment_data = batch['sentiment']
            actual_returns = batch['returns']
            actual_volatility = batch['volatility']
            
            # Forward pass
            predictions, regime_probs, intermediate = model(
                price_data, technical_data, fundamental_data, sentiment_data
            )
            
            # Compute loss
            total_loss, losses = model.compute_total_loss(
                predictions, intermediate, regime_probs,
                actual_returns, actual_volatility
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] += v.item()
        
        # Logging
        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch}: {avg_losses}")
        
        # Learning rate scheduling
        scheduler.step(avg_losses['returns'])
        
        # Validation
        if epoch % config.val_every == 0:
            validate(model, val_loader, config)
    
    return model
```

---

## Summary

This document has covered:

1. **Foundational Concepts**: What PINNs are and why they're powerful for finance
2. **Mathematical Foundations**: PDEs, boundary conditions, autodifferentiation
3. **Architecture Details**: Network design, activation functions, encoding strategies
4. **Financial Equations**: Black-Scholes, Heston, GARCH, Factor Models, Mean-Variance
5. **Loss Engineering**: Multi-component losses, collocation sampling, weighting
6. **Design Patterns**: Multi-output, encoder-decoder, hierarchical, regime-conditional
7. **Training Strategies**: Two-stage training, curriculum learning, optimizers
8. **Trade-offs**: Soft vs hard constraints, data vs physics driven
9. **Practical Tips**: Normalization, noisy data, validation, debugging
10. **Complete Reference Architecture**: The Quant-FINN framework

**Next Steps for Implementation:**
1. Start with a simple Black-Scholes PINN to validate your setup
2. Add GARCH volatility modeling
3. Incorporate factor model structure
4. Add sentiment features
5. Implement regime detection
6. Full Quant-FINN assembly

Good luck with your implementation! 🚀
