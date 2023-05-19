import jax
from jax import numpy as np
from typing import Any, Optional, Tuple, Callable

def get_mapped_fn(fn, group_data):
    group_Xs, group_Ys, group_Ns = group_data
    mapped_fn = (
        lambda model_params: 
        jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_util.tree_map(
                lambda group_X, group_Y, group_N: 
                fn(model_params, group_X, group_Y, group_N), 
                group_Xs, group_Ys, group_Ns
            )
        )
    )
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
    num_groups = len(group_Ns)
    
    loss_fn = (lambda params, group_X, group_Y, group_N: 
        _loss_fn(
            model_fn(params, group_X), group_Y, group_N
        ) / num_groups
    )

    grad_fn = jax.grad(
            lambda params, group_X, group_Y, group_N: 
            loss_fn(params, group_X, group_Y, group_N)
    )

    loss_fn = jax.jit(loss_fn)
    grad_fn = jax.jit(grad_fn)
    
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
        eps: Optional[float] = 1e-6, 
        verbose: Optional[int] = 0, 
        print_every: Optional[int] = 500, 
        lr: Optional[float] = 1e-4,
        mapped_fns = None
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
    if mapped_fns is None:
        loss_fn, grad_fn = get_mapped_fns(loss_fn, model_fn, group_data)
    else:
        loss_fn, grad_fn = mapped_fns

    if verbose == 2:
        "Fitting model..."

    for i in range(maxit):
        grads = grad_fn(model_params)
        grad_norm = np.sqrt(np.sum(grads**2))
        if i % print_every == 0 and verbose == 2:
            print(
                i, loss_fn(model_params), 
                grad_norm, lr
            )
        if np.any(np.isnan(grads)):
            print("NaN gradient update, aborting...")
            break
        model_params = grad_update(model_params, grads, lr=lr)
        if grad_norm < eps and verbose > 0:
            print("Converged!")
            if verbose == 2:
                print(i, loss_fn(model_params), grad_norm, np.max(model_params), lr)
            break

    if grad_norm > eps and verbose > 0:
        print(f"Failed to converge, gradient norm is {grad_norm}.")
        
    return model_params, grad_norm, (loss_fn, grad_fn)
