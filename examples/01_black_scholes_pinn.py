"""
Example 1: Black-Scholes PINN for European Option Pricing

This example demonstrates the basic usage of PINNs for option pricing.
We train a network to solve the Black-Scholes PDE:

    ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

The network learns to:
1. Satisfy the PDE in the interior domain
2. Match the terminal condition V(S,T) = max(S-K, 0) for calls
3. Satisfy boundary conditions at S=0 and S=∞

After training, we compare to the analytical Black-Scholes solution.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')

from src.models.pinn_base import BlackScholesPINN
from src.losses.pde_residuals import compute_black_scholes_residual
from src.training.trainer import PINNTrainer
from src.data.synthetic import generate_black_scholes_data
from src.utils.visualization import (
    plot_solution_surface,
    plot_greeks,
    plot_terminal_condition,
    plot_training_curves,
)
from src.utils.validation import (
    validate_greeks_consistency,
    compare_to_black_scholes,
)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ==========================================
    # Problem Parameters
    # ==========================================
    K = 100.0      # Strike price
    r = 0.05       # Risk-free rate
    sigma = 0.2    # Volatility
    T = 1.0        # Time to expiration

    S_min, S_max = 50.0, 150.0  # Stock price range
    t_min, t_max = 0.0, T       # Time range

    print("\n" + "="*50)
    print("Black-Scholes PINN Example")
    print("="*50)
    print(f"Strike (K): {K}")
    print(f"Risk-free rate (r): {r}")
    print(f"Volatility (sigma): {sigma}")
    print(f"Expiration (T): {T}")
    print(f"Stock price range: [{S_min}, {S_max}]")

    # ==========================================
    # Create Network
    # ==========================================
    print("\n" + "-"*50)
    print("Creating PINN...")

    pinn = BlackScholesPINN(
        hidden_dim=64,
        n_layers=4,
        activation='tanh',
        sigma=sigma,
        r=r,
        use_fourier_encoding=True,
        n_fourier_features=10,
    ).to(device)

    n_params = sum(p.numel() for p in pinn.parameters())
    print(f"Network parameters: {n_params:,}")

    # ==========================================
    # Create Trainer
    # ==========================================
    print("\n" + "-"*50)
    print("Setting up trainer...")

    trainer = PINNTrainer(
        network=pinn,
        pde_residual_fn=compute_black_scholes_residual,
        bounds={'S': (S_min, S_max), 't': (t_min, t_max)},
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type='call',
        lr=1e-3,
        n_collocation=2000,
        n_boundary=500,
        n_terminal=500,
        use_annealing=True,
        warmup_epochs=200,
        sampler_type='lhs',
        device=device,
    )

    # ==========================================
    # Training
    # ==========================================
    print("\n" + "-"*50)
    print("Training PINN...")
    print("This may take a few minutes...")

    history = trainer.train(
        n_epochs=2000,
        verbose=True,
        log_every=200,
    )

    # L-BFGS refinement
    print("\n" + "-"*50)
    trainer.train_with_lbfgs(n_iterations=200, verbose=True)

    # ==========================================
    # Validation
    # ==========================================
    print("\n" + "-"*50)
    print("Validating model...")

    # Generate test points
    n_test = 500
    S_test = torch.linspace(S_min, S_max, n_test, device=device)
    t_test = torch.rand(n_test, device=device) * T

    # Greeks consistency
    greeks_result = validate_greeks_consistency(pinn, S_test, t_test)
    print(f"\nGreeks validation:")
    print(f"  Delta error: {greeks_result['delta']['relative_error']:.4f}")
    print(f"  Gamma error: {greeks_result['gamma']['relative_error']:.4f}")
    print(f"  Theta error: {greeks_result['theta']['relative_error']:.4f}")
    print(f"  Passed: {greeks_result['passed']}")

    # Black-Scholes comparison
    bs_result = compare_to_black_scholes(
        pinn, S_test, t_test, K, r, sigma, T, 'call'
    )
    print(f"\nBlack-Scholes comparison:")
    print(f"  Mean relative error: {bs_result['mean_relative_error']:.4f}")
    print(f"  Max relative error: {bs_result['max_relative_error']:.4f}")
    print(f"  Passed: {bs_result['passed']}")

    # ==========================================
    # Visualization
    # ==========================================
    print("\n" + "-"*50)
    print("Generating plots...")

    # Training curves
    fig1 = plot_training_curves(history)
    fig1.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: training_curves.png")

    # Solution surface
    fig2 = plot_solution_surface(
        pinn,
        S_range=(S_min, S_max),
        t_range=(t_min, t_max),
    )
    fig2.savefig('solution_surface.png', dpi=150, bbox_inches='tight')
    print("Saved: solution_surface.png")

    # Greeks
    fig3 = plot_greeks(pinn, S_range=(S_min, S_max), t_fixed=0.5)
    fig3.savefig('greeks.png', dpi=150, bbox_inches='tight')
    print("Saved: greeks.png")

    # Terminal condition
    fig4 = plot_terminal_condition(
        pinn,
        S_range=(S_min, S_max),
        T=T,
        K=K,
        option_type='call',
    )
    fig4.savefig('terminal_condition.png', dpi=150, bbox_inches='tight')
    print("Saved: terminal_condition.png")

    # ==========================================
    # Example Predictions
    # ==========================================
    print("\n" + "-"*50)
    print("Sample predictions:")

    # Specific points
    sample_S = torch.tensor([80.0, 100.0, 120.0], device=device)
    sample_t = torch.tensor([0.0, 0.0, 0.0], device=device)  # Current time

    pinn.eval()
    with torch.no_grad():
        sample_V = pinn(sample_S, sample_t).squeeze()

    print(f"\n  S       t      V (Network)")
    print(f"  ----    ----   ------------")
    for i in range(len(sample_S)):
        print(f"  {sample_S[i].item():.1f}    {sample_t[i].item():.1f}    {sample_V[i].item():.4f}")

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == '__main__':
    main()
