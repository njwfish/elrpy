import jax
from jax import numpy as np
from jaxopt import BacktrackingLineSearch
from typing import Any, Optional, Tuple, Callable

def get_mapped_loss_and_grad(
        loss_fn: Callable,
        model_fn: Callable,
        group_data: Tuple[Any, Any, Any],
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
    _, _, group_Ns = group_data
    num_groups = len(group_Ns)

    def wrapped_loss(model_params, group_X, group_Y, group_N):
        return loss_fn(model_fn(model_params, group_X), group_Y, group_N) / num_groups

    wrapped_loss = jax.jit(jax.value_and_grad(wrapped_loss))

    def loss_and_grad_fn(model_params, group_data):
        group_Xs, group_Ys, group_Ns = group_data
        num_groups = len(group_Ns)
        group_losses = jax.tree_util.tree_map(
            lambda group_X, group_Y, group_N: 
            wrapped_loss(model_params, group_X, group_Y, group_N), 
            group_Xs, group_Ys, group_Ns
        )
        mean_loss, grad = jax.tree_util.tree_reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1]), group_losses, 
            is_leaf=lambda x: not isinstance(x, dict)
        )
        return mean_loss, grad

    return loss_and_grad_fn



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


def fit(
        loss_fn: Callable, 
        model_fn: Callable,
        model_params: dict, 
        group_data: Tuple[dict],
        maxit: Optional[int] = 500_000, 
        eps: Optional[float] = 1e-6, 
        verbose: Optional[int] = 0, 
        print_every: Optional[int] = 500, 
        init_lr: Optional[float] = 1.0,
        lr_floor: Optional[float] = 1e-6,
        loss_and_grad_fn = None,
        ls = None
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
    n_params = np.prod(np.array(model_params.shape))
    if loss_and_grad_fn is None:
        loss_and_grad_fn = get_mapped_loss_and_grad(loss_fn, model_fn, group_data)
    if ls is None:
        ls = BacktrackingLineSearch(
            fun=loss_and_grad_fn, maxiter=20, 
            condition="goldstein", decrease_factor=0.8, 
            jit=False, value_and_grad=True
        )

    for i in range(maxit):
        loss, grad = loss_and_grad_fn(model_params, group_data)
        grad_norm = np.sqrt(np.sum(grad**2)) / n_params 
        if np.any(np.isnan(grad)):
            print("NaN gradient update, aborting...")
            break
        # stepsize, ls_state = ls.run(
        #     init_stepsize=init_lr, params=model_params,
        #     value=loss, grad=grad,
        #     group_data=group_data
        # )
        stepsize = init_lr
        model_params = grad_update(model_params, grad, lr=max(stepsize, lr_floor))
        if i % print_every == 0 and verbose == 2:
            print(i, "\t", loss, "\t", grad_norm, "\t", stepsize)#, "\t", ls_state.done)
        if grad_norm < eps and verbose > 0:
            print("Converged!")
            if verbose == 2:
                print(i, "\t", loss, "\t", grad_norm, "\t", stepsize)
            break

    if grad_norm > eps and verbose > 0:
        print(f"Failed to converge, gradient norm is {grad_norm}.")
        
    return model_params, grad_norm, loss_and_grad_fn, ls