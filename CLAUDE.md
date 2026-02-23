# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is for implementing **Physics-Informed Neural Networks (PINNs) for Quantitative Finance** - specifically a system called "Quant-FINN". PINNs incorporate domain knowledge (financial PDEs, economic constraints) directly into neural network training, regularizing learning with economic theory and producing outputs consistent with no-arbitrage principles.

## Core Architecture Concept

The Quant-FINN system combines:
- **Data-driven learning** from market observations
- **Physics-informed constraints** from financial mathematics (Black-Scholes, GARCH, factor models)
- **Regime-conditional dynamics** for different market conditions
- **Multi-scale processing** (macro/meso/micro timeframes)

## Key Financial Equations to Embed

### Black-Scholes PDE (Option Pricing)
```
∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```
Inputs: (S, t, σ, r, K) → Output: V (option value)

### GARCH(1,1) Volatility Model
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```
Constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1

### Factor Models (Fama-French)
```
R_i = α_i + Σ_j β_ij · F_j + ε_i
```
Constraint: Residuals must be uncorrelated with factors

### Mean-Variance Portfolio Optimization
```
maximize: μᵀw - (λ/2)wᵀΣw
subject to: Σᵢwᵢ = 1
```

### No-Arbitrage Constraints
- Put-Call Parity: `C - P = S - Ke^(-rT)`
- Monotonicity: `∂C/∂S > 0`, `∂C/∂K < 0`, `∂C/∂σ > 0`

## Critical Implementation Requirements

### Automatic Differentiation
PINNs require computing network derivatives w.r.t. inputs to calculate PDE residuals:
```python
S = S.requires_grad_(True)
V = network(S, t)
dV_dS = torch.autograd.grad(V, S, create_graph=True)[0]
d2V_dS2 = torch.autograd.grad(dV_dS, S, create_graph=True)[0]
```
Always use `create_graph=True` to allow backprop through derivatives.

### Activation Functions
**Must use smooth activations** with well-behaved derivatives:
- `tanh` (standard choice)
- `swish` (x · σ(x))
- `GELU`

**Never use ReLU or LeakyReLU** - their discontinuous derivatives break second-order PDE computations.

### Loss Function Structure
```python
L_total = λ_data · L_data         # Match observed data
        + λ_pde · L_pde           # Satisfy governing PDEs
        + λ_ic · L_ic             # Initial conditions
        + λ_bc · L_bc             # Boundary conditions
        + λ_econ · L_economic     # Economic constraints
```

### Collocation Points
Sample points in input domain where PDE must be satisfied (no labels needed). Strategies:
- Uniform random sampling
- Latin Hypercube Sampling (better coverage)
- Adaptive sampling (more points where residual is high)

### Normalization
Always normalize inputs and outputs. When computing PDE residuals with normalized variables, account for chain rule:
```python
# If V_norm = (V - V_mean) / V_std and S_norm = (S - S_mean) / S_std
# Then: dV/dS = (V_std / S_std) * dV_norm/dS_norm
```

## Network Architecture Guidelines

### Recommended Starting Architecture
- **Depth**: 4-6 hidden layers
- **Width**: 100-200 neurons per layer
- **Activation**: tanh or swish
- **Input encoding**: Fourier features for complex solutions

### Fourier Feature Encoding
Maps inputs to high-dimensional space to overcome spectral bias:
```python
encoded = [sin(2π·f₁·x), cos(2π·f₁·x), ..., sin(2π·fₙ·x), cos(2π·fₙ·x)]
```

### Deep Networks
For 6+ layers, use residual connections:
```python
x = activation(linear2(activation(linear1(x))) + x)  # Skip connection
```

## Training Strategies

### Two-Stage Training
1. **Stage 1**: Train primarily on data (pretrain)
2. **Stage 2**: Add PDE constraints with gradual weight increase

### Loss Weight Annealing
Start with low PDE weight, increase over time:
```python
lambda_pde(epoch) = lambda_initial + (lambda_final - lambda_initial) * (epoch / warmup_epochs)
```

### Optimizers
- **Adam**: Good default (lr=1e-3 to 1e-4)
- **L-BFGS**: For final refinement after Adam training

### Curriculum Learning
Start with easy regions (interior points), gradually include difficult regions (boundaries, singularities).

## Multi-Component Architecture Pattern

### Regime Detection
Classify market state (bull/bear/sideways) and apply regime-specific PDE parameters:
```python
regime_probs = classifier(features)
output = Σᵢ regime_probs[i] · pinn_i(input, params_i)
```

### Multi-Output Heads
Separate heads for related quantities (price, delta, vega) with consistency enforcement:
```python
delta_autodiff = ∂V/∂S  # From autodiff
delta_predicted = delta_head(features)
L_consistency = MSE(delta_autodiff, delta_predicted)
```

### Time Series: Encoder-Decoder
```
Historical Data → LSTM Encoder → Latent (PINN constraints) → Decoder → Predictions
```

## Validation Requirements

1. **Hold out time periods** (not random points)
2. **Validate on different market regimes**
3. **Check PDE residual on unseen collocation points**
4. **Compare Greeks to finite differences**:
```python
delta_fd = (V(S+dS) - V(S-dS)) / (2·dS)
delta_ad = ∂V/∂S  # Should match closely
```

## Common Pitfalls

### PDE Loss Not Decreasing
- Verify autodiff implementation (check `create_graph=True`)
- Verify PDE equation is correct
- Check normalization chain rule

### Training Unstable
- Reduce learning rate
- Normalize inputs/outputs
- Clip gradients

### Poor Extrapolation
- Increase PDE weight
- Add more collocation points in extrapolation region

### Boundary Conditions Not Satisfied
- Increase BC weights
- Consider hard enforcement via architecture

## Key Design Principles

### Soft vs Hard Constraints
- **Soft**: Add to loss function (flexible but may violate)
- **Hard**: Enforce via architecture (e.g., `softplus` for positive prices)

### Data-Driven vs Physics-Driven Balance
- More data → lower `λ_pde` (trust observations)
- Limited data → higher `λ_pde` (trust equations)
- Non-stationary markets → higher `λ_pde` (regularization)

### Network Size
- Simple PDEs: Shallow & wide (2-3 layers, 500+ neurons)
- Complex multi-dimensional: Deep & narrow (6-10 layers, 50-100 neurons)

## Code Organization

Project structure:
```
src/
├── models/
│   ├── pinn_base.py           # Base PINN architecture
│   ├── quant_finn.py           # Full Quant-FINN model
│   ├── encoders.py             # Fourier, LSTM, sentiment encoders
│   └── layers/
│       ├── garch_layer.py      # GARCH dynamics
│       ├── factor_layer.py     # Factor model
│       └── portfolio_layer.py  # Mean-variance optimization
├── losses/
│   ├── pde_residuals.py        # Black-Scholes, Heston residuals
│   ├── economic_constraints.py # No-arbitrage, monotonicity
│   └── loss_weighting.py       # Adaptive weighting strategies
├── training/
│   ├── trainer.py              # Main training loop
│   ├── collocation.py          # Sampling strategies
│   └── curriculum.py           # Curriculum learning
├── data/
│   ├── preprocessing.py        # Normalization, feature engineering
│   └── loaders.py              # Data loading utilities
└── utils/
    ├── autodiff.py             # Helper functions for derivatives
    ├── validation.py           # Greeks validation, residual checks
    └── visualization.py        # Diagnostic plots
```

## Development Workflow

When implementing new components:

1. **Start simple**: Implement basic Black-Scholes PINN first
2. **Validate thoroughly**: Check PDE residuals, Greeks, boundary conditions
3. **Add complexity incrementally**: GARCH → factors → regime detection
4. **Monitor loss components**: Track each term in multi-component loss separately
5. **Visualize diagnostics**: Plot solution surfaces, residuals, delta/gamma

## Testing Strategy

Required validations for PINNs:
- PDE residual < 1e-3 on test collocation points
- Boundary conditions satisfied within 1% error
- Greeks match finite differences within 5%
- No-arbitrage violations < 0.1% of option value
- GARCH parameters satisfy stationarity (α + β < 1)

## Framework Dependencies

Expected PyTorch-based implementation:
- `torch` (autodiff, neural networks)
- `numpy` (numerical operations)
- `scipy` (Latin Hypercube sampling, optimization)
- `matplotlib`/`seaborn` (diagnostics)

Optional:
- `transformers` (FinBERT for sentiment)
- `pytorch-lightning` (training framework)
- `wandb` (experiment tracking)
