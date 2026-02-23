"""
Example 3: Factor Model Layer for Return Prediction

This example demonstrates using the factor model layer to:
1. Learn factor exposures (betas) from data
2. Enforce factor structure as a physics constraint
3. Extract alpha (excess returns)

The factor model:
    R_i = α_i + Σ_j β_{ij} · F_j + ε_i

Enforces APT/Fama-French structure in predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')

from src.models.layers.factor_layer import FactorModelLayer, FamaFrenchLayer
from src.data.synthetic import generate_factor_data


def main():
    # Set seed
    torch.manual_seed(42)
    # Factor model uses matrix operations (not sequential loops),
    # so GPU parallelism provides speedup here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\n" + "="*50)
    print("Factor Model Example")
    print("="*50)

    # ==========================================
    # Generate Synthetic Factor Data
    # ==========================================
    print("\n" + "-"*50)
    print("Generating factor data...")

    n_assets = 10
    n_factors = 3
    n_samples = 1000

    data = generate_factor_data(
        n_samples=n_samples,
        n_assets=n_assets,
        n_factors=n_factors,
        seed=42,
        device=device,
    )

    asset_returns = data['asset_returns']
    factor_returns = data['factor_returns']
    true_betas = data['betas']
    true_alphas = data['alphas']

    print(f"Generated {n_samples} observations for {n_assets} assets")
    print(f"True betas shape: {true_betas.shape}")
    print(f"True alpha mean: {true_alphas.mean().item():.6f}")

    # ==========================================
    # Create and Fit Factor Model Layer
    # ==========================================
    print("\n" + "-"*50)
    print("Fitting factor model...")

    factor_layer = FactorModelLayer(
        n_assets=n_assets,
        n_factors=n_factors,
        hidden_dim=32,
        dynamic_betas=False,  # Use static betas for simplicity
    ).to(device)

    # Training
    optimizer = torch.optim.Adam(factor_layer.parameters(), lr=0.01)
    n_epochs = 500

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        pred_returns, betas, alphas = factor_layer(factor_returns)

        # Prediction loss
        pred_loss = F.mse_loss(pred_returns, asset_returns)

        # Factor structure loss (orthogonality of residuals)
        structure_loss = factor_layer.factor_residual_loss(
            asset_returns, factor_returns
        )

        # Total loss
        loss = pred_loss + 0.1 * structure_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}, "
                  f"pred_loss={pred_loss.item():.6f}, "
                  f"struct_loss={structure_loss.item():.6f}")

    # ==========================================
    # Evaluate Results
    # ==========================================
    print("\n" + "-"*50)
    print("Evaluating learned parameters...")

    learned_betas = factor_layer.get_factor_exposures()

    print("\nBeta comparison (first 3 assets):")
    print("Asset  |  True beta_MKT  |  Learned beta_MKT  |  Error")
    print("-" * 50)
    for i in range(3):
        true_b = true_betas[i, 0].item()
        learned_b = learned_betas[i, 0].item()
        error = abs(true_b - learned_b)
        print(f"  {i}    |    {true_b:.3f}     |     {learned_b:.3f}      |   {error:.3f}")

    # Overall beta error
    beta_mse = F.mse_loss(learned_betas, true_betas)
    print(f"\nOverall beta MSE: {beta_mse.item():.6f}")

    # Alpha comparison
    learned_alphas = factor_layer.alphas.detach()
    alpha_mse = F.mse_loss(learned_alphas, true_alphas)
    print(f"Overall alpha MSE: {alpha_mse.item():.8f}")

    # ==========================================
    # Prediction Quality
    # ==========================================
    print("\n" + "-"*50)
    print("Prediction quality...")

    with torch.no_grad():
        pred_returns, _, _ = factor_layer(factor_returns)

    # R-squared
    ss_res = ((asset_returns - pred_returns) ** 2).sum()
    ss_tot = ((asset_returns - asset_returns.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot

    print(f"R-squared: {r_squared.item():.4f}")

    # Correlation
    for i in range(min(3, n_assets)):
        corr = torch.corrcoef(
            torch.stack([asset_returns[:, i], pred_returns[:, i]])
        )[0, 1]
        print(f"Asset {i} correlation: {corr.item():.4f}")

    # ==========================================
    # Visualization
    # ==========================================
    print("\n" + "-"*50)
    print("Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training loss
    axes[0, 0].plot(losses, 'b-')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_yscale('log')

    # Beta comparison (heatmap)
    im = axes[0, 1].imshow(
        torch.abs(learned_betas - true_betas).cpu().numpy(),
        cmap='Blues',
        aspect='auto'
    )
    axes[0, 1].set_xlabel('Factor')
    axes[0, 1].set_ylabel('Asset')
    axes[0, 1].set_title('|Learned beta - True beta|')
    plt.colorbar(im, ax=axes[0, 1])

    # Actual vs Predicted (asset 0)
    asset_idx = 0
    axes[1, 0].scatter(
        asset_returns[:, asset_idx].cpu().numpy(),
        pred_returns[:, asset_idx].cpu().numpy(),
        alpha=0.3,
        s=10
    )
    min_val = min(asset_returns[:, asset_idx].min(), pred_returns[:, asset_idx].min()).item()
    max_val = max(asset_returns[:, asset_idx].max(), pred_returns[:, asset_idx].max()).item()
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
    axes[1, 0].set_xlabel('Actual Returns')
    axes[1, 0].set_ylabel('Predicted Returns')
    axes[1, 0].set_title(f'Asset {asset_idx}: Actual vs Predicted')
    axes[1, 0].legend()

    # Beta values bar chart
    x = np.arange(n_factors)
    width = 0.35
    for i in range(min(3, n_assets)):
        offset = (i - 1) * width
        axes[1, 1].bar(
            x + offset,
            learned_betas[i].cpu().numpy(),
            width,
            alpha=0.7,
            label=f'Asset {i}'
        )
    axes[1, 1].set_xlabel('Factor')
    axes[1, 1].set_ylabel('Beta')
    axes[1, 1].set_title('Learned Factor Betas')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['MKT', 'F2', 'F3'])
    axes[1, 1].legend()
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('factor_model.png', dpi=150, bbox_inches='tight')
    print("Saved: factor_model.png")

    print("\n" + "="*50)
    print("Factor model example complete!")
    print("="*50)


if __name__ == '__main__':
    main()
