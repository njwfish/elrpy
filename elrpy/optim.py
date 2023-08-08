import jax
from jax import numpy as np
from typing import Any, Optional, Tuple, Callable


def reduce_sum(tree):
    return jax.tree_util.tree_reduce(
        jax.jit(lambda x, y: x + y if isinstance(x, np.ndarray) else [xi + yi for (xi, yi) in zip(x, y)]), 
        tree, is_leaf=lambda x: not isinstance(x, dict)
    )


def get_mapped_fn(fn):
    def mapped_fn(model_params, args):
        group_values = jax.tree_util.tree_map(
            lambda *args: fn(model_params, *args), *args
        )
        return reduce_sum(group_values)
    return mapped_fn


def get_wrapped_loss(loss_fn, model_fn, num_groups):
    def wrapped_loss(model_params, group_X, group_Y, group_N, group_weight=None):
        return loss_fn(
            model_fn(model_params, group_X), group_Y, group_N, weights=group_weight
        ) / num_groups
    
    return wrapped_loss

def get_cg_fn(grad_fn, hess_fn):
    def cg_fn(params, *args):
        loss, grad = grad_fn(params, *args) 
        hess = hess_fn(params, *args)
        return loss, np.linalg.solve(hess, grad)
    return cg_fn

def clip_inv_prod(eps, grad, u, v):
    inv_hess = v @ np.diag(1 / np.maximum(u, eps)) @ v.T
    next_cg = inv_hess @ grad
    return next_cg

eigh =  jax.jit(np.linalg.eigh)
clip_inv_prod = jax.vmap(clip_inv_prod, in_axes=(0, None, None, None))

def get_clipped_cg_fn(loss_fn, grad_fn, hess_fn, start_clip=1e-1, stop_clip=1e-4, num_clip=30):
    eps = np.logspace(
        np.log(start_clip), np.log(stop_clip), num_clip, base=np.exp(1)
    )
    def cg_fn(params, *args):
        _, grad = grad_fn(params, *args) 
        hess = hess_fn(params, *args)
        u, v = eigh(hess)
        cg = clip_inv_prod(eps, grad, u, v).T
        losses = loss_fn(params[:, None] - cg, *args)
        min_idx = np.argmin(losses)
        print(grad @ cg[:, min_idx], np.linalg.norm(grad))
        return losses[min_idx], cg[:, min_idx], np.linalg.norm(grad)
    return cg_fn


def gd(
        loss_fn: Callable, 
        model_fn: Callable,
        model_params: dict, 
        group_data: Tuple[dict],
        maxit: Optional[int] = 1_000, 
        tol: Optional[float] = 1e-8, 
        verbose: Optional[int] = 0, 
        print_every: Optional[int] = 50, 
        save_every: Optional[int] = 50,
        save_dir: Optional[str] = None,
        lr: Optional[float] = 1.0,
        mapped_loss_and_dir_fn = None,
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
        loss_and_grad_fn: loss and gradient function
        group_weights: (bootstrap) weights to weight groups by
    
    Returns:
        fitted model parameters
        gradient norm
    """
    if mapped_loss_and_dir_fn is None:
        print("Gradient functions not provided, these will be recompiled...")
        num_groups = len(group_data[0])
        loss_fn = get_wrapped_loss(loss_fn, model_fn, num_groups)
        grad_fn = jax.jit(jax.value_and_grad(loss_fn))
        mapped_loss_and_dir_fn = get_mapped_fn(grad_fn)

    if group_weights is not None:
        group_data = (group_data[0], group_data[1], group_data[2], group_weights)

    for i in range(maxit):
        out = mapped_loss_and_dir_fn(model_params, group_data)
        if len(out) == 2:
            loss, grad = out
            grad_norm = np.linalg.norm(grad)
        elif len(out) == 3:
            loss, grad, grad_norm = out
        else:
            raise ValueError("Unexpected number of outputs from mapped loss and gradient function.")

        if np.any(np.isnan(grad)):
            print("NaN gradient update, aborting...")
            break
        
        if i % print_every == 0 and verbose == 2:
            print(i, "\t", loss, "\t", grad_norm)
        if grad_norm < tol and verbose > 0:
            print("Converged!")
            if verbose == 2:
                print(i, loss, grad_norm)
            break

        model_params -= lr * grad
        if i % save_every == 0 and save_dir is not None:
            np.savez(f"{save_dir}/model.npz", model_params=model_params, grad_norm=grad_norm, i=i)

    if grad_norm > tol and verbose > 0:
        print(f"Failed to converge, gradient norm is {grad_norm}.")
    if save_dir is not None:
        np.savez(f"{save_dir}/model.npz", model_params=model_params, grad_norm=grad_norm, i=i)
        
    return model_params, grad_norm


