import jax
from jax import numpy as np
from typing import Any, Optional, Tuple, Callable


def get_bfgs_fn(grad_fn, start=1., stop=1e-8, num=10, c_1=1e-3, c_2=0.1):
    alpha = np.logspace(
        np.log(start), np.log(stop), num, base=np.exp(1)
    )
    def bfgs_fn(params, state, *args, loss=None, grad=None):
        if len(state) == 3:
            hess, loss, grad = state
        else:
            hess = state
        if loss is None or grad is None:
            loss, grad = grad_fn(params[None, :], *args) 
            loss, grad = loss[0], grad[0]
        dir = -hess @ grad
        losses, grads = grad_fn(params + alpha[:, None] * dir[None, :], *args)
        armijo = losses <= loss + c_1 * alpha * np.linalg.norm(grad) ** 2
        wolfe = grads @ dir >= c_2 * grad @ dir
        idx = np.where(wolfe & armijo & np.all(~np.isnan(grads), axis=1))[0]
        if len(idx) != 0:
            idx = idx[0]
            update = alpha[idx] * dir
            new_loss, new_grad = losses[idx], grads[idx]
        else:
            update = -1e-4 * grad
            new_loss, new_grad = grad_fn(params[None, :] + update, *args) 
            new_loss, new_grad = new_loss[0], new_grad[0]


        diff = new_grad - grad
        hess += (diff @ update + diff @ hess @ diff) * (update @ update) / (update @ diff) ** 2 
        hess -= (hess @ np.outer(diff, update) + np.outer(update, diff) @ hess) / (update @ diff)
        return loss, -update, (hess, new_loss, new_grad), np.linalg.norm(grad)
    return bfgs_fn

def get_bfgs_fn2(grad_fn, start=1., stop=1e-8, num=10, c_1=1e-3, c_2=0.1):
    alpha = np.logspace(
        np.log(start), np.log(stop), num, base=np.exp(1)
    )
    def bfgs_fn(params, state, *args, loss=None, grad=None):
        if len(state) == 3:
            hess, loss, grad = state
        else:
            hess = state
        if loss is None or grad is None:
            loss, grad = mapped_loss_and_grad_fn(params, *args) 
        # print(params)
        dir = -hess @ grad
        # losses, grads = grad_fn(params + alpha[:, None] * dir[None, :], *args)
        # armijo = losses <= loss + c_1 * alpha * np.linalg.norm(grad) ** 2
        # wolfe = grads @ dir >= c_2 * grad @ dir
        # idx = np.where(wolfe & armijo & np.all(~np.isnan(grads), axis=1))[0]
        # print(loss)
        alpha, state = ls.run(
            init_stepsize=1e-2, params=params,
            descent_direction=dir
        )
        print(alpha)
        if state.done:
            print("success")
            update = alpha * dir
            # new_loss, new_grad = losses[idx], grads[idx]
        else:
            update = -1e-4 * grad

        new_loss, new_grad = mapped_loss_and_grad_fn(params + update, *args) 
        diff = new_grad - grad
        hess += (diff @ update + diff @ hess @ diff) * (update @ update) / (update @ diff) ** 2 
        hess -= (hess @ np.outer(diff, update) + np.outer(update, diff) @ hess) / (update @ diff)
        return loss, -update, (hess, new_loss, new_grad), np.linalg.norm(grad)
    return bfgs_fn


def bfgs(
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
        bfgs_fn = None,
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
    hess = np.eye(model_params.shape[0])
    for i in range(maxit):
        loss, dir, hess, grad_norm = bfgs_fn(model_params, hess, group_data)

        if np.any(np.isnan(dir)):
            print("NaN gradient update, aborting...")
            break
        
        if i % print_every == 0 and verbose == 2:
            print(i, "\t", loss, "\t", grad_norm)
        if grad_norm < tol and verbose > 0:
            print("Converged!")
            if verbose == 2:
                print(i, loss, grad_norm)
            break

        model_params -= dir
        if i % save_every == 0 and save_dir is not None:
            np.savez(f"{save_dir}/model.npz", model_params=model_params, grad_norm=grad_norm, i=i)

    if grad_norm > tol and verbose > 0:
        print(f"Failed to converge, gradient norm is {grad_norm}.")
    if save_dir is not None:
        np.savez(f"{save_dir}/model.npz", model_params=model_params, grad_norm=grad_norm, i=i)
        
    return model_params, grad_norm