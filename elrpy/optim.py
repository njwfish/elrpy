import jax
from jax import numpy as np
from typing import Any, Optional, Tuple, Callable

def get_mapped_loss_and_grad(
        loss_fn: Callable,
        model_fn: Callable,
        num_groups: int
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
    
    def wrapped_loss(model_params, group_X, group_Y, group_N, group_weight=None):
        return loss_fn(model_fn(model_params, group_X), group_Y, group_N, weights=group_weight) / num_groups

    wrapped_grad_fn = jax.jit(jax.grad(wrapped_loss))
    wrapped_loss_and_grad_fn = jax.jit(jax.value_and_grad(wrapped_loss))

    def loss_and_grad_fn(model_params, group_data, group_weights=None):
        group_Xs, group_Ys, group_Ns = group_data
        if group_weights is not None:
            group_losses = jax.tree_util.tree_map(
                lambda group_X, group_Y, group_N, group_weight: 
                wrapped_loss_and_grad_fn(
                    model_params, group_X, group_Y, group_N, group_weight=group_weight
                ), 
                group_Xs, group_Ys, group_Ns, group_weights
            )
        else:
            group_losses = jax.tree_util.tree_map(
                lambda group_X, group_Y, group_N: 
                wrapped_loss_and_grad_fn(
                    model_params, group_X, group_Y, group_N
                ), 
                group_Xs, group_Ys, group_Ns
            )
        mean_loss, grad = jax.tree_util.tree_reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1]), group_losses, 
            is_leaf=lambda x: not isinstance(x, dict)
        )
        return mean_loss, grad

    return wrapped_grad_fn, wrapped_loss_and_grad_fn, loss_and_grad_fn



def grad_update(params: np.ndarray, grads: np.ndarray, lr: Optional[float]=1.0) -> dict:
    """Update parameters using gradient descent.
    
    Args:
        params: dict of parameters
        grads: dict of gradients
        lr: learning rate
        
    Returns:
        updated parameters
    """
    return params - lr * grads


def gd(
        loss_fn: Callable, 
        model_fn: Callable,
        model_params: dict, 
        group_data: Tuple[dict],
        maxit: Optional[int] = 500_000, 
        tol: Optional[float] = 1e-8, 
        verbose: Optional[int] = 0, 
        print_every: Optional[int] = 500, 
        lr: Optional[float] = 1.0,
        loss_and_grad_fn = None,
        group_weights=None
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
    if loss_and_grad_fn is None:
        num_groups = len(group_data[0])
        _, _, loss_and_grad_fn = get_mapped_loss_and_grad(loss_fn, model_fn, num_groups)

    for i in range(maxit):
        loss, grad = loss_and_grad_fn(model_params, group_data, group_weights=group_weights)
        grad_norm = np.sqrt(np.sum(grad**2))
        if np.any(np.isnan(grad)):
            print("NaN gradient update, aborting...")
            break
        
        model_params = grad_update(model_params, grad, lr=lr)
        if i % print_every == 0 and verbose == 2:
            print(i, "\t", loss, "\t", grad_norm)
        if grad_norm < tol and verbose > 0:
            print("Converged!")
            if verbose == 2:
                print(i, loss, grad_norm)
            break

    if grad_norm > tol and verbose > 0:
        print(f"Failed to converge, gradient norm is {grad_norm}.")
        
    return model_params, grad_norm

def sgd(
        rng,
        loss_fn: Callable, 
        model_fn: Callable,
        model_params: dict, 
        group_data: Tuple[dict],
        maxit: Optional[int] = 500_000, 
        tol: Optional[float] = 1e-8, 
        verbose: Optional[int] = 0, 
        print_every: Optional[int] = 500, 
        lr: Optional[float] = 0.5,
        grad_fn = None,
        loss_and_grad_fn=None,
        group_weights=None
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
    group_Xs, group_Ys, group_Ns = group_data
    num_groups = len(group_Ys)
    
    if grad_fn is None or loss_and_grad_fn is None:
        print("Gradient functions not provided, these will be recompiled...")
        grad_fn, _, loss_and_grad_fn = get_mapped_loss_and_grad(loss_fn, model_fn, num_groups)
    
    groups = list(group_Ys.keys())
    
    for i in range(maxit):
        rng, next_rng = jax.random.split(rng)
        groups = [groups[i] for i in jax.random.permutation(next_rng, len(groups))]
        for g in groups:
            group_weight = group_weights[g] if group_weights is not None else None
            grads = grad_fn(model_params, group_Xs[g], group_Ys[g], group_Ns[g], group_weight=group_weight)
            if np.any(np.isnan(grads)):
                print("NaN gradient update, aborting...")
                return model_params, np.inf
            model_params = grad_update(model_params, grads, lr=num_groups * lr)
        
        if i % print_every == 0 and verbose == 2:
            loss, grads = loss_and_grad_fn(model_params, group_data)
            grad_norm = np.sqrt(np.sum(grads**2))
            print(i, loss, grad_norm)
        if grad_norm < tol and verbose > 0:
            print("Converged!")
            if verbose == 2:
                loss, grads = loss_and_grad_fn(model_params, group_data)
                grad_norm = np.sqrt(np.sum(grads**2))
                print(i, loss, grad_norm)
            return model_params, grad_norm

    print(f"Failed to converge, gradient norm is {grad_norm}.")
    return model_params, grad_norm