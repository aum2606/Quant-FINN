"""
Example 2: GARCH Volatility Modeling with Physics-Informed Layer

This example demonstrates using the GARCH layer for volatility forecasting.
The GARCH(1,1) model captures:
- Volatility clustering
- Mean reversion
- Persistence

We use the GARCH layer as a physics-informed constraint, ensuring
predicted volatility follows GARCH dynamics.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')

from src.models.layers.garch_layer import GARCHLayer, GARCHRegimeLayer
from src.data.synthetic import generate_garch_data


def main():
    # Set seed
    torch.manual_seed(42)
    # Note: GARCH training uses a sequential for-loop over 2000 timesteps.
    # This is much faster on CPU than GPU because GPU kernel launch overhead
    # dominates for small sequential operations. Use CPU for this example.
    device = 'cpu'
    print(f"Using device: {device}")

    print("\n" + "="*50)
    print("GARCH Volatility Example")
    print("="*50)

    # ==========================================
    # Generate Synthetic GARCH Data
    # ==========================================
    print("\n" + "-"*50)
    print("Generating GARCH data...")

    # True GARCH parameters
    true_omega = 0.00001
    true_alpha = 0.1
    true_beta = 0.85

    data = generate_garch_data(
        n_samples=2000,
        omega=true_omega,
        alpha=true_alpha,
        beta=true_beta,
        seed=42,
        device=device,
    )

    returns = data['returns']
    true_variance = data['variance']
    true_volatility = data['volatility']

    print(f"Generated {len(returns)} observations")
    print(f"True parameters: omega={true_omega:.6f}, alpha={true_alpha:.2f}, beta={true_beta:.2f}")
    print(f"True persistence: {true_alpha + true_beta:.3f}")

    # ==========================================
    # Create and Fit GARCH Layer
    # ==========================================
    print("\n" + "-"*50)
    print("Fitting GARCH layer...")

    # Initialize with rough estimates
    garch = GARCHLayer(
        omega=0.0001,
        alpha=0.15,
        beta=0.80,
        learnable=True,
    ).to(device)

    # Training setup
    optimizer = torch.optim.Adam(garch.parameters(), lr=0.01)
    n_epochs = 500

    # Add batch dimension
    returns_batch = returns.unsqueeze(0)

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Compute predicted variances
        pred_variance = garch(returns_batch)

        # GARCH consistency loss + likelihood loss
        # Simple proxy: MSE between squared returns and predicted variance
        # (In practice, you'd use proper quasi-likelihood)
        target = returns_batch.pow(2)
        loss = nn.functional.mse_loss(pred_variance, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}, "
                  f"omega={garch.omega.item():.6f}, "
                  f"alpha={garch.alpha.item():.3f}, "
                  f"beta={garch.beta.item():.3f}")

    print("\n" + "-"*50)
    print("Learned parameters:")
    print(f"  omega: {garch.omega.item():.6f} (true: {true_omega:.6f})")
    print(f"  alpha: {garch.alpha.item():.3f} (true: {true_alpha:.2f})")
    print(f"  beta: {garch.beta.item():.3f} (true: {true_beta:.2f})")
    print(f"  Persistence: {garch.persistence.item():.3f} (true: {true_alpha + true_beta:.2f})")

    # ==========================================
    # Volatility Forecasting
    # ==========================================
    print("\n" + "-"*50)
    print("Forecasting volatility...")

    # Get in-sample predictions
    with torch.no_grad():
        pred_variance = garch(returns_batch).squeeze()
        pred_volatility = torch.sqrt(pred_variance + 1e-8)

    # Multi-step forecast
    current_var = pred_variance[-1]
    forecast_horizon = 20
    forecasts = garch.multi_step_forecast(
        current_var.unsqueeze(0),
        n_steps=forecast_horizon,
    ).squeeze()

    forecast_vol = torch.sqrt(forecasts + 1e-8)

    print(f"Current volatility: {pred_volatility[-1].item():.4f}")
    print(f"1-step forecast: {forecast_vol[0].item():.4f}")
    print(f"10-step forecast: {forecast_vol[9].item():.4f}")
    print(f"Long-run volatility: {torch.sqrt(garch.unconditional_variance).item():.4f}")

    # ==========================================
    # Visualization
    # ==========================================
    print("\n" + "-"*50)
    print("Generating plots...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Returns
    axes[0].plot(returns.cpu().numpy(), 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_ylabel('Returns')
    axes[0].set_title('Simulated Returns')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Volatility comparison
    t = np.arange(len(returns))
    axes[1].plot(t, true_volatility.cpu().numpy(), 'b-', label='True', alpha=0.7)
    axes[1].plot(t, pred_volatility.cpu().numpy(), 'r--', label='Predicted', alpha=0.7)
    axes[1].set_ylabel('Volatility')
    axes[1].set_title('True vs Predicted Volatility')
    axes[1].legend()

    # Training loss
    axes[2].plot(losses, 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig('garch_volatility.png', dpi=150, bbox_inches='tight')
    print("Saved: garch_volatility.png")

    # ==========================================
    # Regime-Switching GARCH Demo
    # ==========================================
    print("\n" + "-"*50)
    print("Regime-switching GARCH demo...")

    regime_garch = GARCHRegimeLayer(n_regimes=2).to(device)

    # Simulate regime probabilities (in practice, these come from a classifier)
    regime_probs = torch.zeros(1, 2, device=device)
    regime_probs[0, 0] = 0.7  # 70% low volatility regime
    regime_probs[0, 1] = 0.3  # 30% high volatility regime

    with torch.no_grad():
        regime_variance = regime_garch(returns_batch, regime_probs).squeeze()

    print(f"Regime-weighted variance computed")
    print(f"Regime 0 params: {regime_garch.get_regime_parameters()['regime_0']}")
    print(f"Regime 1 params: {regime_garch.get_regime_parameters()['regime_1']}")

    print("\n" + "="*50)
    print("GARCH example complete!")
    print("="*50)


if __name__ == '__main__':
    main()
