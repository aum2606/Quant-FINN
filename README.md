# Quant-FINN: Physics-Informed Neural Networks for Quantitative Finance

A PyTorch implementation of Physics-Informed Neural Networks (PINNs) for financial applications — option pricing, volatility modeling, and factor-based return prediction.

PINNs embed domain knowledge (financial PDEs, no-arbitrage constraints) directly into the neural network's training loss. This produces models that respect economic theory by design, not by accident.

---

## Features

| Module | Description |
|--------|-------------|
| **Black-Scholes PINN** | Solves the BS PDE via autodiff; computes Delta, Gamma, Theta |
| **GARCH Layer** | Differentiable GARCH(1,1) with learnable parameters and regime-switching |
| **Factor Model Layer** | Fama-French factor structure with residual orthogonality constraints |
| **Adaptive Training** | Loss annealing, Latin Hypercube collocation sampling, L-BFGS refinement |
| **Visualization** | Solution surfaces, Greeks plots, terminal condition comparison |

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd financeneuralnet

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Or activate (Linux/macOS)
source venv/bin/activate
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.0+, NumPy, SciPy, Pandas, Matplotlib, Seaborn, PyYAML, tqdm

For GPU acceleration, install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org).

---

## Quick Start

### Option Pricing with Black-Scholes PINN

```python
import torch
from src.models.pinn_base import BlackScholesPINN
from src.losses.pde_residuals import compute_black_scholes_residual
from src.training.trainer import PINNTrainer

# Create network
pinn = BlackScholesPINN(
    hidden_dim=64,
    n_layers=4,
    sigma=0.2,
    r=0.05,
    use_fourier_encoding=True,
)

# Train with PDE constraints
trainer = PINNTrainer(
    network=pinn,
    pde_residual_fn=compute_black_scholes_residual,
    bounds={'S': (50, 150), 't': (0, 1)},
    K=100.0, T=1.0, r=0.05, sigma=0.2,
    n_collocation=2000,
    sampler_type='lhs',
)
history = trainer.train(n_epochs=2000)

# Predict and compute Greeks
S = torch.tensor([80.0, 100.0, 120.0])
t = torch.tensor([0.0, 0.0, 0.0])
V = pinn(S, t)
delta, gamma, theta, _ = pinn.compute_greeks(S, t)
```

### GARCH Volatility Modeling

```python
from src.models.layers.garch_layer import GARCHLayer

# Learnable GARCH(1,1) layer
garch = GARCHLayer(omega=0.0001, alpha=0.1, beta=0.85, learnable=True)

returns = torch.randn(1, 1000)          # (batch, sequence)
variances = garch(returns)              # Conditional variances
forecasts = garch.multi_step_forecast(variances[:, -1:], n_steps=20)
```

### Factor Model

```python
from src.models.layers.factor_layer import FactorModelLayer

factor_layer = FactorModelLayer(n_assets=10, n_factors=3)
pred_returns, betas, alphas = factor_layer(factor_returns)

# Physics constraint: residuals must be uncorrelated with factors
loss = pred_loss + 0.1 * factor_layer.factor_residual_loss(actual, factors)
```

---

## Running Examples

> **Note:** Run all commands from the project root directory.

**Windows PowerShell:**
```powershell
$env:PYTHONPATH = "C:\path\to\financeneuralnet"
$env:PYTHONIOENCODING = "utf-8"
$env:MPLBACKEND = "Agg"

# Example 1: Black-Scholes PINN (GPU, ~2 min)
& .\venv\Scripts\python.exe examples\01_black_scholes_pinn.py

# Example 2: GARCH Volatility (CPU, ~30s)
& .\venv\Scripts\python.exe examples\02_garch_volatility.py

# Example 3: Factor Model (GPU, ~5s)
& .\venv\Scripts\python.exe examples\03_factor_model.py
```

**Linux/macOS:**
```bash
export PYTHONPATH=$(pwd)
python examples/01_black_scholes_pinn.py
python examples/02_garch_volatility.py
python examples/03_factor_model.py
```

Each example saves diagnostic plots as PNG files in the project root.

---

## Project Structure

```
financeneuralnet/
├── examples/
│   ├── 01_black_scholes_pinn.py    # Option pricing demo
│   ├── 02_garch_volatility.py      # Volatility modeling demo
│   └── 03_factor_model.py          # Factor model demo
├── src/
│   ├── models/
│   │   ├── pinn_base.py            # Base PINN + BlackScholesPINN
│   │   ├── quant_finn.py           # Full Quant-FINN model
│   │   ├── encoders.py             # Fourier, LSTM encoders
│   │   └── layers/
│   │       ├── garch_layer.py      # GARCH(1,1) + regime-switching
│   │       ├── factor_layer.py     # Factor model (Fama-French, APT)
│   │       └── portfolio_layer.py  # Mean-variance optimization
│   ├── losses/
│   │   ├── pde_residuals.py        # Black-Scholes, Heston PDE losses
│   │   ├── economic_constraints.py # No-arbitrage, monotonicity
│   │   └── loss_weighting.py       # Fixed, annealing, adaptive weights
│   ├── training/
│   │   ├── trainer.py              # Main training loop (Adam + L-BFGS)
│   │   └── collocation.py          # Uniform, LHS, adaptive sampling
│   ├── data/
│   │   ├── synthetic.py            # GBM, Heston, GARCH, factor data
│   │   └── preprocessing.py        # Normalization, feature engineering
│   └── utils/
│       ├── autodiff.py             # Gradient/Hessian helpers
│       ├── validation.py           # Greeks, PDE residual, BS comparison
│       └── visualization.py        # Solution surface, Greeks, loss plots
├── configs/
│   └── default_config.yaml         # All hyperparameters
├── tests/
│   └── test_pinn_base.py           # Unit tests
├── requirements.txt
├── setup.py
└── CLAUDE.md                       # Architecture and implementation guide
```

---

## Key Concepts

### Physics-Informed Loss

The total loss balances data fit against physical constraints:

```
L_total = λ_data · L_data  +  λ_pde · L_pde  +  λ_bc · L_boundary  +  λ_econ · L_economic
```

Weights are annealed during training — starting data-dominant, gradually weighting physics more.

### Black-Scholes PDE Constraint

```
∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

Enforced at collocation points sampled via Latin Hypercube Sampling (no labels needed):

```python
S = S.requires_grad_(True)
V = network(S, t)
dV_dS  = torch.autograd.grad(V, S, create_graph=True)[0]
d2V_dS2 = torch.autograd.grad(dV_dS, S, create_graph=True)[0]
residual = dV_dt + 0.5*sigma**2*S**2*d2V_dS2 + r*S*dV_dS - r*V
```

### GARCH Dynamics

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Constraints enforced via parameter reparameterization: `α + β < 1` (stationarity), `ω > 0`.

### Factor Model Structure

```
R_i = α_i + Σ_j β_ij · F_j + ε_i
```

The `factor_residual_loss` penalizes correlation between residuals ε and factors F, enforcing factor orthogonality as a physics constraint.

---

## Configuration

Edit `configs/default_config.yaml` to tune:
- **Network:** depth, width, activation (`tanh`, `swish`, `gelu`), Fourier encoding
- **Training:** learning rate, epochs, loss weights, sampling strategy
- **Problem:** strike K, volatility σ, risk-free rate r, time horizon T

---

## Testing

```bash
# Run unit tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems.* Journal of Computational Physics.
- Tancik, M., et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* NeurIPS.
- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.* Journal of Econometrics.
- Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.* Journal of Financial Economics.

---

## License

MIT License
