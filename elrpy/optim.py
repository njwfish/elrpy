import jax
from jax import numpy as np
from typing import Any, Optional, Tuple, Callable

def get_cg_fn(loss_and_grad_fn, hess_fn, l2=0.0):
    """Return standard conjugate gradient function, using lstsq to solve the linear system in case of ill-conditioning.

    Args:
        grad_fn: function returning loss and gradient.
        hess_fn: Hessian function.
        l2: L2 regularization strength.

    Returns:
        function: conjugate gradient function.
    """
    def cg_fn(params, *args):
        """Conjugate gradient function using lstsq to solve the linear system in case of ill-conditioning.

        Args:
            params: model parameters.
            *args: arguments to pass to loss and gradient function.

        Returns:
            tuple: loss, conjugate gradient, gradient norm.
        """
        loss, grad = loss_and_grad_fn(params, *args) 
        hess = hess_fn(params, *args)
        loss += 0.5 * l2 * np.sum(params ** 2)
        grad += l2 * params
        hess += l2 * np.eye(hess.shape[-1])
        return np.mean(loss), np.linalg.lstsq(hess, grad)[0], np.linalg.norm(grad, axis=-1)
    return cg_fn

def clip_inv_prod(eps, grad, u, v):
    """Clip eigenvalues and compute inverse product.
    
    Args:
        eps: clipping values.
        grad: gradient.
        u: eigenvalues.
        v: eigenvectors.
        
    Returns:
        np.ndarray: inverse product.
    """
    inv_hess = v @ np.diag(1 / np.maximum(u, eps)) @ v.T
    next_cg = inv_hess @ grad
    return next_cg

eigh =  jax.jit(np.linalg.eigh)
clip_inv_prod = jax.jit(jax.vmap(clip_inv_prod, in_axes=(0, None, None, None)))

def get_clipped_cg_fn(loss_fn, grad_fn, hess_fn, start_clip=1e-1, stop_clip=1e-4, num_clip=10, l2=0.0):
    """Return conjugate gradient function with eigenvalue clipping in case of ill-conditioning.
    Eigenvalues are clipped between start_clip and stop_clip, with num_clip values in between,
    with the clipped inverse product computed using the eigendecomposition of the Hessian.
    The conjugate gradient is then computed as the minimum of the clipped inverse products and the
    standard conjugate gradient.

    Args:
        loss_fn: loss function.
        grad_fn: gradient function.
        hess_fn: Hessian function.
        start_clip: starting clipping value.
        stop_clip: stopping clipping value.
        num_clip: number of clipping values.
        l2: L2 regularization strength.

    Returns:
        function: conjugate gradient function.
    """

    # compute clipping values
    eps = np.logspace(
        np.log(start_clip), np.log(stop_clip), num_clip, base=np.exp(1)
    )

    # define vectorizing helper functions
    def cp(params, grad, hess):
        """Compute conjugate gradient using clipped inverse product.
        
        Args:
            params: model parameters.
            grad: gradient.
            hess: Hessian.
            
        Returns:
            tuple: gradient, Hessian, updated parameters, conjugate gradient.
        """
        grad += l2 * params
        hess += l2 * np.eye(hess.shape[0])
        u, v = eigh(hess)
        cg = clip_inv_prod(eps, grad, u, v).T
        return grad, hess, params[:, None] - cg, cg
    
    def gl(losses, cg, params, grad):
        """Compute loss, conjugate gradient and gradient norm.

        Args:
            losses: losses.
            cg: conjugate gradient.
            params: model parameters.
            grad: gradient.

        Returns:
            tuple: losses, conjugate gradient, gradient norm.
        """
        min_idx = np.argmin(losses + 0.5 * l2 * np.sum((params[:, None] - cg) ** 2, axis=0), axis=0)
        return losses[min_idx], cg[:, min_idx], np.linalg.norm(grad)

    # vectorize helper functions
    # cp = jax.jit(jax.vmap(cp, in_axes=(0, 0, 0)))
    # gl = jax.jit(jax.vmap(gl, in_axes=(0, 0, 0, 0)))

    def cg_fn(params, *args):
        """Conjugate gradient function using clipped inverse product.

        Args:
            params: model parameters.
            *args: arguments to pass to loss and gradient function.

        Returns:
            tuple: loss, conjugate gradient, gradient norm.
        """
        loss, grad = grad_fn(params, *args) 
        hess = hess_fn(params, *args)
        grad, hess, updated_params, cg = cp(params, grad, hess)
        losses = jax.vmap(loss_fn, in_axes=(1, *([None] * len(args))))(updated_params, *args) 
        losses, cg, grad_norm = gl(losses, cg, params, grad)
        return loss, cg, np.max(grad_norm)
    return cg_fn

def fit(
        loss_fn: Callable, 
        model_fn: Callable,
        model_params: dict, 
        group_data: Tuple[dict],
        lr: Optional[float] = 1.0,
        maxit: Optional[int] = 1_000, 
        tol: Optional[float] = 1e-8, 
        verbose: Optional[int] = 0, 
        print_every: Optional[int] = 50, 
        save_every: Optional[int] = 50,
        save_dir: Optional[str] = None,
        mapped_loss_and_dir_fn = None,
        group_weights=None,
        keep_history=False,
        dir_type='cg'
) -> Tuple[dict, float]:
    """Fit a model to data using directional descent with the given loss and direction functions.

    Args:
        loss_fn: loss function.
        model_fn: model function.
        model_params: model parameters.
        group_data: tuple of group covariates, group outcomes and group number of observations.
        maxit: maximum number of iterations.
        tol: tolerance for convergence.
        verbose: verbosity level (0: silent, 1: standard prints, 2: debug prints).
        print_every: print every n iterations.
        save_every: save model every n iterations.
        save_dir: directory to save model.
        lr: learning rate.
        mapped_loss_and_dir_fn: mapped loss and direction function, if None, will be compiled from loss function.
        group_weights: weights for each group.
        keep_history: whether to keep history of model parameters, loss and gradient norm.

    Returns:
        tuple: model parameters, gradient norm.
    """
    if mapped_loss_and_dir_fn is None:
        from elrpy.losses import get_wrapped_loss
        if verbose > 0:
            print("Gradient functions not provided, these will be recompiled...")
        wrapped_loss_fn = get_wrapped_loss(loss_fn, model_fn)
        if dir_type == 'cg':
            if group_weights is None:
                _loss_fn = wrapped_loss_fn
                loss_and_grad_fn = jax.value_and_grad(wrapped_loss_fn)
                hess_fn = jax.hessian(wrapped_loss_fn)
            else:
                _loss_fn = jax.vmap(wrapped_loss_fn, in_axes=(0, None, None, None))
                loss_and_grad_fn = jax.vmap(jax.value_and_grad(wrapped_loss_fn), in_axes=(0, None, None, None))
                hess_fn = jax.vmap(jax.hessian(wrapped_loss_fn), in_axes=(0, None, None, None))
            mapped_loss_and_dir_fn = jax.jit(get_clipped_cg_fn(_loss_fn, loss_and_grad_fn, hess_fn))
        elif dir_type == 'gd':
            mapped_loss_and_dir_fn = jax.jit(jax.value_and_grad(wrapped_loss_fn))

    if group_weights is not None:
        group_data = (group_data[0], group_data[1], group_data[2], group_weights)

    if keep_history:
        history = []

    for i in range(maxit):
        out = mapped_loss_and_dir_fn(model_params, *group_data)
        if len(out) == 2:
            loss, grad = out
            grad_norm = np.linalg.norm(grad, axis=-1)
        elif len(out) == 3:
            loss, grad, grad_norm = out
        else:
            raise ValueError("Unexpected number of outputs from mapped loss and gradient function.")
        
        if keep_history:
            history.append((model_params, loss, grad_norm))

        if np.any(np.isnan(grad)):
            if verbose > 0:
                print(f"NaN gradient update, aborting with loss {loss}...")
            break
        
        if i % print_every == 0 and verbose == 2:
            print(i, "\t", loss, "\t", grad_norm)
        if np.all(grad_norm < tol) and verbose > 0:
            print("Converged!")
            if verbose == 2:
                print(i, loss, grad_norm)
            break

        model_params -= (grad_norm >= tol) * lr * grad
        if i % save_every == 0 and save_dir is not None:
            np.savez(f"{save_dir}/model.npz", model_params=model_params, grad_norm=grad_norm, i=i)

    if grad_norm > tol and verbose > 0:
        print(f"Failed to converge, gradient norm is {grad_norm}.")
    if save_dir is not None:
        np.savez(f"{save_dir}/model.npz", model_params=model_params, grad_norm=grad_norm, i=i)

    if keep_history:
        params, losses, grad_norms = (np.array(z) for z in zip(*history))
        history = (params, losses, grad_norms)
    
    return (model_params, grad_norm, history) if keep_history else (model_params, grad_norm)
