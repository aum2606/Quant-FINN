"""
Unit Tests for PINN Base Module.

Tests the fundamental PINN architecture components:
- Network forward pass
- Autodiff for derivatives
- PDE residual computation
- Boundary conditions
"""

import pytest
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from src.models.pinn_base import PINNBase, BlackScholesPINN, FourierEncoder
from src.utils.autodiff import compute_gradient, compute_second_derivative


class TestFourierEncoder:
    """Tests for Fourier feature encoding."""

    def test_output_shape(self):
        """Test that encoder produces correct output shape."""
        encoder = FourierEncoder(input_dim=2, n_frequencies=5)
        x = torch.randn(32, 2)

        output = encoder(x)

        # Expected: 2 * 2 * 5 (sin + cos) + 2 (raw input) = 22
        expected_dim = 2 * 2 * 5 + 2
        assert output.shape == (32, expected_dim)

    def test_deterministic(self):
        """Test that encoding is deterministic."""
        encoder = FourierEncoder(input_dim=2, n_frequencies=5)
        x = torch.randn(10, 2)

        out1 = encoder(x)
        out2 = encoder(x)

        assert torch.allclose(out1, out2)

    def test_output_dim_property(self):
        """Test output_dim property."""
        encoder = FourierEncoder(input_dim=3, n_frequencies=10, include_input=True)

        expected = 3 * 2 * 10 + 3
        assert encoder.output_dim == expected


class TestPINNBase:
    """Tests for base PINN architecture."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dim=32, n_layers=3)
        x = torch.randn(16, 2)

        output = pinn(x[:, 0], x[:, 1])

        assert output.shape == (16, 1)

    def test_gradient_flow(self):
        """Test that gradients flow through network."""
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dim=32, n_layers=3)
        x = torch.randn(16, 2, requires_grad=True)

        output = pinn(x[:, 0], x[:, 1])
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_fourier_encoding_option(self):
        """Test network with Fourier encoding enabled."""
        pinn = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dim=32,
            n_layers=3,
            use_fourier_encoding=True,
            n_fourier_features=5,
        )
        x = torch.randn(16, 2)

        output = pinn(x[:, 0], x[:, 1])

        assert output.shape == (16, 1)


class TestBlackScholesPINN:
    """Tests for Black-Scholes PINN."""

    @pytest.fixture
    def pinn(self):
        """Create a Black-Scholes PINN for testing."""
        return BlackScholesPINN(
            hidden_dim=32,
            n_layers=3,
            sigma=0.2,
            r=0.05,
        )

    def test_forward_pass(self, pinn):
        """Test forward pass with S and t inputs."""
        S = torch.linspace(50, 150, 32)
        t = torch.rand(32)

        V = pinn(S, t)

        assert V.shape == (32, 1)
        assert not torch.isnan(V).any()

    def test_pde_residual_shape(self, pinn):
        """Test PDE residual computation shape."""
        S = torch.linspace(50, 150, 32).requires_grad_(True)
        t = torch.rand(32).requires_grad_(True)

        residual = pinn.pde_residual(S, t)

        assert residual.shape == (32,)

    def test_pde_residual_differentiable(self, pinn):
        """Test that PDE residual is differentiable."""
        S = torch.linspace(50, 150, 32).requires_grad_(True)
        t = torch.rand(32).requires_grad_(True)

        residual = pinn.pde_residual(S, t)
        loss = residual.pow(2).mean()
        loss.backward()

        # Check gradients exist for network parameters
        for param in pinn.parameters():
            assert param.grad is not None

    def test_greeks_computation(self, pinn):
        """Test Greeks (delta, gamma, theta) computation."""
        S = torch.linspace(80, 120, 16).requires_grad_(True)
        t = torch.full((16,), 0.5).requires_grad_(True)

        delta, gamma, theta, V = pinn.compute_greeks(S, t)

        assert delta.shape == (16,)
        assert gamma.shape == (16,)
        assert theta.shape == (16,)
        assert V.shape == (16,)

        # Delta should be in [0, 1] for calls (approximately)
        # Allow some slack for untrained network
        assert delta.min() > -1.0
        assert delta.max() < 2.0

    def test_terminal_condition_loss(self, pinn):
        """Test terminal condition loss computation."""
        S = torch.linspace(50, 150, 32)
        K = 100.0

        loss = pinn.terminal_condition_loss(S, K, option_type='call')

        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_boundary_condition_loss(self, pinn):
        """Test boundary condition loss computation."""
        t = torch.rand(32)

        loss = pinn.boundary_condition_loss(t, S_min=0, S_max=200, K=100, T=1.0)

        assert loss.shape == ()
        assert loss.item() >= 0


class TestAutodiff:
    """Tests for autodiff utilities."""

    def test_compute_gradient(self):
        """Test first derivative computation."""
        x = torch.linspace(0, 1, 10).requires_grad_(True)
        y = x ** 2  # dy/dx = 2x

        dy_dx = compute_gradient(y, x)

        expected = 2 * x
        assert torch.allclose(dy_dx, expected, atol=1e-5)

    def test_compute_second_derivative(self):
        """Test second derivative computation."""
        x = torch.linspace(0, 1, 10).requires_grad_(True)
        y = x ** 3  # d²y/dx² = 6x

        d2y_dx2 = compute_second_derivative(y, x)

        expected = 6 * x
        assert torch.allclose(d2y_dx2, expected, atol=1e-4)

    def test_gradient_through_network(self):
        """Test gradient computation through a neural network."""
        net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        x = torch.linspace(0, 1, 20).unsqueeze(-1).requires_grad_(True)
        y = net(x).squeeze()

        dy_dx = compute_gradient(y, x)

        assert dy_dx.shape == x.shape
        assert not torch.isnan(dy_dx).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
