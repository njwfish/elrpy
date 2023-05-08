import jax
from jax import numpy as np
from typing import Any, Optional, Tuple, Callable

def get_mapped_fn(fn, group_data):
    group_Xs, group_Ys, group_Ns = group_data
    num_groups = len(group_Ns)
    mapped_fn = (lambda model_params: 
    jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(
            lambda group_X, group_Y, group_N: 
            fn(model_params, group_X, group_Y, group_N), 
            group_Xs, group_Ys, group_Ns
        )
    ) / num_groups)
    return mapped_fn

def get_mapped_fns(
        _loss_fn: Callable,
        model_fn: Callable,
        group_data: Tuple[dict]
) -> Tuple[Callable, Callable]:
    """Returns a function that computes the gradient of the loss function with
    respect to the model parameters.

    Args:
        _loss_fn (function): loss function
        model_fn (function): model function
        data (tuple): tuple of 

    Returns:
        tuple: loss function, gradient function
    """
    group_Xs, group_Ys, group_Ns = group_data
    
    loss_fn = jax.jit(
        lambda params, group_X, group_Y, group_N: 
        _loss_fn(
            model_fn(params, group_X), group_Y, group_N
        )
    )

    grad_fn = jax.jit(
        jax.grad(
            lambda params, group_X, group_Y, group_N: 
            loss_fn(params, group_X, group_Y, group_N)
        )
    )
    
    mapped_loss_fn = get_mapped_fn(
        loss_fn, (group_Xs, group_Ys, group_Ns)
    )
    mapped_grad_fn = get_mapped_fn(
        grad_fn, (group_Xs, group_Ys, group_Ns)
    )

    return mapped_loss_fn, mapped_grad_fn



def grad_update(params: dict, grads: dict, lr: Optional[float]=1.0) -> dict:
    """Update parameters using gradient descent.
    
    Args:
        params: dict of parameters
        grads: dict of gradients
        lr: learning rate
        
    Returns:
        updated parameters
    """
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def fit(
        loss_fn: Callable, 
        model_fn: Callable,
        model_params: dict, 
        group_data: Tuple[dict],
        maxit: Optional[int] = 500_000, 
        eps: Optional[float] = 1e-8, 
        verbose: Optional[bool] = False, 
        lr: Optional[float] = 0.5
) -> Tuple[dict, float]:
    """Fit a model to data using gradient descent.

    Args:
        loss_fn: loss function
        model_fn: model function taking model parameters and data as arguments
        model_params: model parameters
        maxit: maximum number of iterations
        eps: convergence threshold
        verbose: whether to print progress
        lr: learning rate
    
    Returns:
        fitted model parameters
        gradient norm
    """
    loss_fn, grad_fn = get_mapped_fns(loss_fn, model_fn, group_data)

    for i in range(maxit):
        grads = {"beta": grad_fn(model_params)}
        grad_norm = sum(np.sum(grads[k]**2) for k in grads)
        if i % 500 == 0 and verbose:
            print(i, loss_fn(model_params,), grad_norm)
        if any([np.any(np.isnan(grads[k])) for k in grads]):
            print("NaN gradient update, aborting...")
            break
        model_params = grad_update(model_params, grads, lr=lr)
        if grad_norm < eps:
            print("Converged!")
            break

    if grad_norm > eps:
        print(f"Failed to converge, gradient norm is {grad_norm}.")

    return model_params, grad_norm