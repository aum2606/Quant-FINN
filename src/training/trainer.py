"""
PINN Training Loop Implementation.

This module provides a complete training loop for PINNs with:
- Multi-component loss (data + PDE + boundary + economic)
- Flexible loss weighting strategies
- Adaptive collocation sampling
- Learning rate scheduling
- Comprehensive logging

The trainer follows a two-stage approach:
1. Warmup: Focus on data fit with low PDE weight
2. Main training: Gradually increase physics constraints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Dict, Optional, List, Callable, Any
from collections import defaultdict
import time
from tqdm import tqdm

from src.training.collocation import (
    UniformSampler,
    LatinHypercubeSampler,
    AdaptiveSampler,
    BoundaryConditionSampler,
)
from src.losses.loss_weighting import (
    FixedWeighting,
    AnnealingScheduler,
    AdaptiveWeighting,
    LossTracker,
)


class PINNTrainer:
    """
    Physics-Informed Neural Network Trainer.

    Handles the complete training loop including:
    - Multi-component loss computation
    - Collocation point sampling
    - Loss weighting (fixed, annealing, or adaptive)
    - Optimization (Adam + optional L-BFGS refinement)
    - Logging and checkpointing

    Example:
        >>> trainer = PINNTrainer(
        ...     network=pinn,
        ...     pde_residual_fn=black_scholes_residual,
        ...     bounds={'S': (50, 150), 't': (0, 1)},
        ...     K=100.0,
        ...     T=1.0,
        ... )
        >>> trainer.train(n_epochs=1000)
    """

    def __init__(
        self,
        network: nn.Module,
        pde_residual_fn: Callable,
        bounds: Dict[str, tuple],
        K: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        sigma: float = 0.2,
        option_type: str = 'call',
        # Training config
        lr: float = 1e-3,
        n_collocation: int = 1000,
        n_boundary: int = 200,
        n_terminal: int = 200,
        # Loss weights
        loss_weights: Optional[Dict[str, float]] = None,
        use_annealing: bool = True,
        warmup_epochs: int = 100,
        # Sampling
        sampler_type: str = 'lhs',
        adaptive_sampling: bool = False,
        # Device
        device: str = 'cpu',
    ):
        """
        Initialize PINN trainer.

        Args:
            network: Neural network to train.
            pde_residual_fn: Function computing PDE residual.
            bounds: Variable bounds for collocation sampling.
            K: Strike price.
            T: Expiration time.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            lr: Learning rate.
            n_collocation: Number of interior collocation points.
            n_boundary: Points per boundary.
            n_terminal: Points for terminal condition.
            loss_weights: Initial loss weights.
            use_annealing: Whether to anneal PDE weight.
            warmup_epochs: Epochs before full PDE weight.
            sampler_type: 'uniform', 'lhs', or 'importance'.
            adaptive_sampling: Whether to use adaptive sampling.
            device: Training device.
        """
        self.network = network.to(device)
        self.pde_residual_fn = pde_residual_fn
        self.bounds = bounds
        self.device = device

        # Problem parameters
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type

        # Training config
        self.lr = lr
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_terminal = n_terminal

        # Initialize sampler
        self._init_sampler(sampler_type, adaptive_sampling)

        # Initialize boundary sampler
        S_bounds = bounds.get('S', (0, 200))
        t_bounds = bounds.get('t', (0, 1))
        self.boundary_sampler = BoundaryConditionSampler(
            S_bounds, t_bounds, device
        )

        # Initialize loss weighting
        self._init_weighting(loss_weights, use_annealing, warmup_epochs)

        # Optimizer
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            min_lr=1e-6,
        )

        # Tracking
        self.loss_tracker = LossTracker([
            'pde', 'terminal', 'lower_bc', 'upper_bc', 'data'
        ])
        self.epoch = 0

    def _init_sampler(self, sampler_type: str, adaptive: bool):
        """Initialize collocation point sampler."""
        if adaptive:
            self.sampler = AdaptiveSampler(self.bounds, self.n_collocation, self.device)
        elif sampler_type == 'lhs':
            self.sampler = LatinHypercubeSampler(self.bounds, self.device)
        else:
            self.sampler = UniformSampler(self.bounds, self.device)

        self.adaptive_sampling = adaptive

    def _init_weighting(
        self,
        weights: Optional[Dict[str, float]],
        use_annealing: bool,
        warmup_epochs: int,
    ):
        """Initialize loss weighting strategy."""
        default_weights = {
            'pde': 1.0,
            'terminal': 10.0,
            'lower_bc': 10.0,
            'upper_bc': 10.0,
            'data': 1.0,
        }

        if weights:
            default_weights.update(weights)

        if use_annealing:
            initial = default_weights.copy()
            initial['pde'] = 0.01  # Start with low PDE weight

            self.weighting = AnnealingScheduler(
                initial_weights=initial,
                final_weights=default_weights,
                warmup_epochs=warmup_epochs,
            )
        else:
            self.weighting = FixedWeighting(default_weights)

    def compute_losses(
        self,
        data_points: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute all loss components.

        Args:
            data_points: Optional observed data points.

        Returns:
            Dict of loss components.
        """
        losses = {}

        # 1. PDE Loss (interior collocation points)
        collocation = self.sampler.sample(self.n_collocation)
        S_col = collocation['S'].requires_grad_(True)
        t_col = collocation['t'].requires_grad_(True)

        pde_residual = self.pde_residual_fn(
            self.network, S_col, t_col,
            sigma=self.sigma, r=self.r
        )
        losses['pde'] = pde_residual.pow(2).mean()

        # Store residual for adaptive sampling
        self._last_residual = pde_residual.detach()
        self._last_collocation = collocation

        # 2. Terminal Condition Loss
        terminal = self.boundary_sampler.sample_terminal(self.n_terminal)
        S_term = terminal['S']
        t_term = terminal['t']

        V_term = self.network(S_term, t_term).squeeze()

        if self.option_type == 'call':
            payoff = torch.relu(S_term - self.K)
        else:
            payoff = torch.relu(self.K - S_term)

        losses['terminal'] = nn.functional.mse_loss(V_term, payoff)

        # 3. Lower Boundary Loss (S = 0)
        lower_bc = self.boundary_sampler.sample_lower_boundary(self.n_boundary)
        V_lower = self.network(lower_bc['S'], lower_bc['t']).squeeze()

        # For calls: V(0, t) = 0
        # For puts: V(0, t) = K * exp(-r*(T-t))
        if self.option_type == 'call':
            expected_lower = torch.zeros_like(V_lower)
        else:
            tau = self.T - lower_bc['t']
            expected_lower = self.K * torch.exp(-self.r * tau)

        losses['lower_bc'] = nn.functional.mse_loss(V_lower, expected_lower)

        # 4. Upper Boundary Loss (S = S_max)
        upper_bc = self.boundary_sampler.sample_upper_boundary(self.n_boundary)
        S_upper = upper_bc['S']
        t_upper = upper_bc['t']
        V_upper = self.network(S_upper, t_upper).squeeze()

        # For calls: V(S_max, t) ≈ S_max - K*exp(-r*(T-t))
        # For puts: V(S_max, t) ≈ 0
        tau = self.T - t_upper
        if self.option_type == 'call':
            expected_upper = S_upper - self.K * torch.exp(-self.r * tau)
        else:
            expected_upper = torch.zeros_like(V_upper)

        losses['upper_bc'] = nn.functional.mse_loss(V_upper, expected_upper)

        # 5. Data Loss (if provided)
        if data_points is not None:
            S_data = data_points['S']
            t_data = data_points['t']
            V_data = data_points['V']

            V_pred = self.network(S_data, t_data).squeeze()
            losses['data'] = nn.functional.mse_loss(V_pred, V_data)
        else:
            losses['data'] = torch.tensor(0.0, device=self.device)

        return losses

    def train_step(
        self,
        data_points: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            data_points: Optional observed data.

        Returns:
            Dict of loss values.
        """
        self.network.train()
        self.optimizer.zero_grad()

        # Compute losses
        losses = self.compute_losses(data_points)

        # Apply weighting
        total_loss = self.weighting(losses, self.epoch)

        # Backward pass
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        # Convert to float for logging
        loss_values = {k: v.item() for k, v in losses.items()}
        loss_values['total'] = total_loss.item()

        return loss_values

    def train(
        self,
        n_epochs: int,
        data_points: Optional[Dict[str, Tensor]] = None,
        verbose: bool = True,
        log_every: int = 100,
        checkpoint_every: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            n_epochs: Number of training epochs.
            data_points: Optional observed data.
            verbose: Whether to print progress.
            log_every: Epochs between logging.
            checkpoint_every: Epochs between checkpoints.
            checkpoint_path: Path for saving checkpoints.

        Returns:
            Dict of loss histories.
        """
        history = defaultdict(list)
        start_time = time.time()

        # Progress bar
        pbar = tqdm(range(n_epochs), disable=not verbose, desc="Training")

        for epoch in pbar:
            self.epoch = epoch

            # Training step
            loss_values = self.train_step(data_points)

            # Update tracking
            for key, value in loss_values.items():
                history[key].append(value)

            self.loss_tracker.update(
                {k: torch.tensor(v) for k, v in loss_values.items() if k != 'total'},
                torch.tensor(loss_values['total'])
            )

            # Learning rate scheduling
            self.scheduler.step(loss_values['total'])

            # Adaptive sampling update
            if self.adaptive_sampling and epoch > 0 and epoch % 100 == 0:
                self.sampler.update(
                    self._last_residual.abs(),
                    n_new_points=100,
                )

            # Logging
            if verbose and epoch % log_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss_values['total']:.2e}",
                    'pde': f"{loss_values['pde']:.2e}",
                    'term': f"{loss_values['terminal']:.2e}",
                    'lr': f"{lr:.1e}",
                })

            # Checkpointing
            if checkpoint_every and checkpoint_path and epoch % checkpoint_every == 0:
                self.save_checkpoint(f"{checkpoint_path}_epoch{epoch}.pt")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {elapsed:.1f}s")
            print(f"Final loss: {history['total'][-1]:.4e}")

        return dict(history)

    def train_with_lbfgs(
        self,
        n_iterations: int = 500,
        data_points: Optional[Dict[str, Tensor]] = None,
        verbose: bool = True,
    ):
        """
        Fine-tune with L-BFGS optimizer.

        L-BFGS is a quasi-Newton method that often achieves better
        final convergence than Adam for PINNs.

        Args:
            n_iterations: Maximum L-BFGS iterations.
            data_points: Optional observed data.
            verbose: Whether to print progress.
        """
        if verbose:
            print("\nStarting L-BFGS refinement...")

        optimizer = optim.LBFGS(
            self.network.parameters(),
            lr=1.0,
            max_iter=n_iterations,
            history_size=50,
            line_search_fn='strong_wolfe',
        )

        iteration = [0]

        def closure():
            optimizer.zero_grad()
            losses = self.compute_losses(data_points)
            total = sum(losses.values())
            total.backward()

            iteration[0] += 1
            if verbose and iteration[0] % 50 == 0:
                print(f"  L-BFGS iteration {iteration[0]}: loss = {total.item():.4e}")

            return total

        optimizer.step(closure)

        if verbose:
            final_losses = self.compute_losses(data_points)
            final_total = sum(v.item() for v in final_losses.values())
            print(f"L-BFGS completed. Final loss: {final_total:.4e}")

    def evaluate(
        self,
        S: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Evaluate network on given points.

        Args:
            S: Stock prices.
            t: Times.

        Returns:
            Option values.
        """
        self.network.eval()
        with torch.no_grad():
            return self.network(S.to(self.device), t.to(self.device))

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'loss_history': self.loss_tracker.get_history(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])

    def get_loss_history(self) -> Dict[str, List[float]]:
        """Get loss history for plotting."""
        return self.loss_tracker.get_history()
