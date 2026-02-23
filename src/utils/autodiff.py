"""
Automatic Differentiation Utilities for PINNs.

This module provides helper functions for computing derivatives of neural network
outputs with respect to inputs. These derivatives are essential for computing
PDE residuals in physics-informed neural networks.

Key Concept:
-----------
PINNs require computing partial derivatives like ∂V/∂S, ∂²V/∂S² to evaluate
whether the network output satisfies the governing PDE. PyTorch's autograd
enables this through computational graph differentiation.

CRITICAL: Always use create_graph=True when you need to backpropagate through
the derivative computation (which is almost always the case in PINNs).
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


def compute_gradient(
    output: Tensor,
    input_var: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
    allow_unused: bool = False,
) -> Tensor:
    """
    Compute the gradient of a scalar output with respect to an input variable.

    This is the fundamental operation for PINNs - computing ∂u/∂x where u is
    the network output and x is an input variable.

    Args:
        output: Network output tensor. Shape can be (batch,) or (batch, 1).
        input_var: Input variable to differentiate with respect to.
                   Must have requires_grad=True.
        create_graph: If True, graph of the derivative will be constructed,
                      allowing computation of higher order derivatives.
                      MUST be True for PINN training.
        retain_graph: If True, retain the computational graph for subsequent
                      backward passes. Usually True for PINNs.
        allow_unused: If True, allows computing gradients even if input_var
                      doesn't affect output.

    Returns:
        Gradient tensor with same shape as input_var.

    Example:
        >>> S = torch.tensor([100.0, 110.0], requires_grad=True)
        >>> V = network(S)  # Option prices
        >>> dV_dS = compute_gradient(V, S)  # Delta (∂V/∂S)

    Note:
        The input_var MUST have requires_grad=True before the forward pass.
        Setting it after won't work because the computational graph won't
        track operations on that variable.
    """
    # Ensure output is the right shape for grad computation
    if output.dim() == 2 and output.shape[1] == 1:
        output = output.squeeze(-1)

    # grad_outputs specifies the "upstream" gradient - we use ones for sum
    grad_outputs = torch.ones_like(output)

    gradient = torch.autograd.grad(
        outputs=output,
        inputs=input_var,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )[0]

    return gradient


def compute_second_derivative(
    output: Tensor,
    input_var: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute the second derivative ∂²u/∂x² of output with respect to input.

    This is essential for PDEs like Black-Scholes which contain terms like
    ½σ²S²(∂²V/∂S²). We compute it by differentiating the first derivative.

    Args:
        output: Network output tensor.
        input_var: Input variable (must have requires_grad=True).
        create_graph: Must be True if you need to backprop through this.
        retain_graph: Whether to retain graph for subsequent computations.

    Returns:
        Second derivative tensor (same shape as input_var).

    Example:
        >>> S = torch.tensor([100.0], requires_grad=True)
        >>> V = network(S)
        >>> gamma = compute_second_derivative(V, S)  # Γ = ∂²V/∂S²
    """
    # First derivative
    first_deriv = compute_gradient(
        output, input_var,
        create_graph=True,  # Need graph for second derivative
        retain_graph=True
    )

    # Second derivative (derivative of first derivative)
    second_deriv = compute_gradient(
        first_deriv, input_var,
        create_graph=create_graph,
        retain_graph=retain_graph
    )

    return second_deriv


def compute_hessian_diagonal(
    output: Tensor,
    input_vars: Tuple[Tensor, ...],
    create_graph: bool = True,
) -> Tuple[Tensor, ...]:
    """
    Compute diagonal elements of the Hessian matrix.

    For a multi-input function u(x₁, x₂, ..., xₙ), this computes
    (∂²u/∂x₁², ∂²u/∂x₂², ..., ∂²u/∂xₙ²).

    This is more efficient than computing the full Hessian when you only
    need diagonal elements, which is common in financial PDEs.

    Args:
        output: Network output tensor.
        input_vars: Tuple of input tensors to differentiate with respect to.
        create_graph: Whether to create graph for backprop.

    Returns:
        Tuple of second derivatives, one for each input variable.

    Example:
        >>> S = torch.tensor([100.0], requires_grad=True)
        >>> t = torch.tensor([0.5], requires_grad=True)
        >>> V = network(S, t)
        >>> d2V_dS2, d2V_dt2 = compute_hessian_diagonal(V, (S, t))
    """
    hessian_diag = []

    for var in input_vars:
        second_deriv = compute_second_derivative(
            output, var,
            create_graph=create_graph,
            retain_graph=True
        )
        hessian_diag.append(second_deriv)

    return tuple(hessian_diag)


def compute_jacobian(
    output: Tensor,
    input_var: Tensor,
    create_graph: bool = True,
) -> Tensor:
    """
    Compute the Jacobian matrix for vector-valued output.

    If output has shape (batch, m) and input has shape (batch, n),
    returns Jacobian of shape (batch, m, n) where J[b,i,j] = ∂output[b,i]/∂input[b,j].

    This is useful for multi-output PINNs where you need to compute
    derivatives of each output with respect to each input.

    Args:
        output: Network output tensor of shape (batch, m).
        input_var: Input tensor of shape (batch, n).
        create_graph: Whether to create graph for backprop.

    Returns:
        Jacobian tensor of shape (batch, m, n).

    Note:
        This implementation uses a loop over output dimensions.
        For large output dimensions, consider using torch.autograd.functional.jacobian.
    """
    batch_size = output.shape[0]
    output_dim = output.shape[1] if output.dim() > 1 else 1
    input_dim = input_var.shape[1] if input_var.dim() > 1 else 1

    jacobian = torch.zeros(batch_size, output_dim, input_dim, device=output.device)

    for i in range(output_dim):
        # Extract i-th output component
        output_i = output[:, i] if output.dim() > 1 else output

        # Compute gradient with respect to all inputs
        grad_i = torch.autograd.grad(
            outputs=output_i,
            inputs=input_var,
            grad_outputs=torch.ones_like(output_i),
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        if grad_i.dim() == 1:
            grad_i = grad_i.unsqueeze(-1)

        jacobian[:, i, :] = grad_i

    return jacobian


def compute_mixed_derivative(
    output: Tensor,
    var1: Tensor,
    var2: Tensor,
    create_graph: bool = True,
) -> Tensor:
    """
    Compute mixed partial derivative ∂²u/(∂x∂y).

    This is needed for PDEs with cross-derivative terms, such as the
    Heston model which has ρξvS(∂²V/∂S∂v).

    Args:
        output: Network output tensor.
        var1: First variable to differentiate with respect to.
        var2: Second variable to differentiate with respect to.
        create_graph: Whether to create graph for backprop.

    Returns:
        Mixed partial derivative tensor.

    Example:
        >>> S = torch.tensor([100.0], requires_grad=True)
        >>> v = torch.tensor([0.04], requires_grad=True)  # variance
        >>> V = network(S, v)
        >>> d2V_dSdv = compute_mixed_derivative(V, S, v)
    """
    # First derivative with respect to var1
    first_deriv = compute_gradient(
        output, var1,
        create_graph=True,
        retain_graph=True
    )

    # Second derivative with respect to var2
    mixed_deriv = compute_gradient(
        first_deriv, var2,
        create_graph=create_graph,
        retain_graph=True
    )

    return mixed_deriv


def ensure_requires_grad(*tensors: Tensor) -> Tuple[Tensor, ...]:
    """
    Ensure all input tensors have requires_grad=True.

    This is a convenience function to prepare inputs for PINN computations.
    Creates new tensors with gradient tracking enabled.

    Args:
        *tensors: Variable number of tensors to process.

    Returns:
        Tuple of tensors with requires_grad=True.

    Example:
        >>> S, t, sigma = ensure_requires_grad(S, t, sigma)
        >>> V = network(S, t, sigma)
        >>> dV_dS = compute_gradient(V, S)  # Now works
    """
    result = []
    for tensor in tensors:
        if not tensor.requires_grad:
            tensor = tensor.clone().detach().requires_grad_(True)
        result.append(tensor)
    return tuple(result)
