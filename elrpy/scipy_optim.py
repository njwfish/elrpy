"""SciPy optimizers for sparse ELR objectives.

The ELR objectives are cheap in parameter dimension but expensive in data size:
each objective evaluation pushes voter-level probabilities through a sparse
precinct aggregation matrix. The production optimizer uses a deterministic
first-order coarse solve followed by a float64 quasi-Newton polish. The coarse
solve avoids expensive curvature work when the zero start is badly conditioned;
the polish keeps optimization deterministic without compiling Hessian-vector
products through the full voter-level computation graph on large states.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from jax.experimental import sparse as jsparse
from scipy import sparse as ssparse
from scipy.optimize import OptimizeResult, minimize

HessianMode = Literal["dense", "hvp", "none"]
TrustMode = Literal["fast", "polished", "strict"]


@dataclass(frozen=True)
class ScipyStage:
    """One SciPy optimization stage."""

    method: str = "trust-ncg"
    hessian: HessianMode = "hvp"
    dtype: str = "float32"
    gtol: float = 2e-2
    maxiter: int = 100
    options: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ScipyFitResult:
    """Summary of a SciPy optimizer run."""

    params: onp.ndarray
    loss: float
    grad_norm: float
    success: bool
    message: str
    nit: int
    nfev: int
    njev: int
    nhev: Optional[int]
    seconds: float
    scipy_result: OptimizeResult
    stage: Optional[ScipyStage] = None
    accepted_by_loss_stability: bool = False


@dataclass(frozen=True)
class MultiStageFitResult:
    """Summary of a staged trust-region fit."""

    params: onp.ndarray
    loss: float
    grad_norm: float
    success: bool
    seconds: float
    stages: tuple[ScipyFitResult, ...]


def default_trust_stages(mode: TrustMode = "polished") -> tuple[ScipyStage, ...]:
    """Return the production trust-region schedule for a named mode.

    `fast` is intended for quick stable fitting. `polished` first gets into the
    right basin with a float32 L-BFGS-B solve and then performs a float64 BFGS
    polish for final reported optima. `strict` uses a tighter final gradient
    tolerance.
    """

    if mode == "fast":
        return (
            ScipyStage(
                method="L-BFGS-B",
                hessian="none",
                dtype="float32",
                gtol=2e-2,
                maxiter=200,
                options={"maxls": 50},
            ),
        )
    if mode == "polished":
        return (
            ScipyStage(
                method="L-BFGS-B",
                hessian="none",
                dtype="float32",
                gtol=2e-2,
                maxiter=200,
                options={"maxls": 50},
            ),
            ScipyStage(
                method="BFGS",
                hessian="none",
                dtype="float64",
                gtol=1e-5,
                maxiter=500,
                options={"c2": 0.9},
            ),
        )
    if mode == "strict":
        return (
            ScipyStage(
                method="L-BFGS-B",
                hessian="none",
                dtype="float32",
                gtol=2e-2,
                maxiter=200,
                options={"maxls": 50},
            ),
            ScipyStage(
                method="BFGS",
                hessian="none",
                dtype="float64",
                gtol=1e-6,
                maxiter=1000,
                options={"c2": 0.9},
            ),
        )
    raise ValueError("mode must be one of 'fast', 'polished', or 'strict'")


def _as_bcoo(matrix, dtype):
    if isinstance(matrix, jsparse.BCOO):
        return jsparse.BCOO(
            (jnp.asarray(matrix.data, dtype=dtype), matrix.indices),
            shape=matrix.shape,
        )
    if ssparse.issparse(matrix):
        return jsparse.BCOO.from_scipy_sparse(matrix.astype(onp.dtype(dtype).name))
    return jnp.asarray(matrix, dtype=dtype)


def _scipy_dtype(dtype) -> onp.dtype:
    if dtype is None:
        return onp.dtype(onp.float64)
    return onp.dtype(dtype)


def _require_x64_if_needed(stages: Sequence[ScipyStage]) -> None:
    needs_x64 = any(onp.dtype(stage.dtype) == onp.dtype(onp.float64) for stage in stages)
    if needs_x64 and not bool(jax.config.jax_enable_x64):
        raise RuntimeError(
            "A float64 optimization stage requires `jax_enable_x64=True`. "
            "Set `jax.config.update('jax_enable_x64', True)` before creating "
            "JAX arrays or before calling the polished/strict optimizer."
        )


def _no_meaningful_gradient_descent(
    objective: Callable[[onp.ndarray], tuple[float, onp.ndarray]],
    theta: onp.ndarray,
    loss: float,
    grad: onp.ndarray,
    *,
    loss_rtol: float = 1e-10,
) -> bool:
    """Return true when `-grad` has no numerically meaningful descent."""

    direction = -onp.asarray(grad, dtype=onp.float64)
    threshold = loss_rtol * max(1.0, abs(float(loss)))
    for step in (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        trial_loss, _ = objective(theta + step * direction)
        if trial_loss < loss - threshold:
            return False
    return True


def fit_scipy(
    loss_fn: Callable,
    model_fn: Callable,
    model_params,
    group_data: Tuple[Any, Any, Any],
    *,
    method: str = "trust-ncg",
    hessian: HessianMode = "hvp",
    dtype=None,
    gtol: float = 1e-5,
    maxiter: int = 100,
    options: Optional[dict[str, Any]] = None,
    stage: Optional[ScipyStage] = None,
) -> ScipyFitResult:
    """Fit a sparse ELR objective with a standard SciPy optimizer.

    Parameters
    ----------
    loss_fn, model_fn, model_params, group_data:
        Same objects accepted by `elrpy.optim.fit`. `group_data` should be
        `(X, Y, G)`, where `G` can be a JAX `BCOO`, a SciPy sparse matrix, or a
        dense array.
    method:
        Any SciPy method compatible with `jac=True`. The production stages use
        first-order methods by default; `trust-ncg` remains available for
        targeted experiments through `hessian="hvp"`.
    hessian:
        `"dense"` passes a dense JAX Hessian to SciPy. `"hvp"` passes Hessian-
        vector products. `"none"` passes only value and gradient, appropriate
        for BFGS/L-BFGS-B.
    dtype:
        Numeric dtype used inside JAX. Use `onp.float32` for fast coarse solves
        and `onp.float64` for high-precision polishing. If using float64, enable
        `jax_enable_x64` before constructing arrays in the calling process.
    gtol, maxiter, options:
        SciPy optimizer controls. Explicit `options` override `gtol`/`maxiter`.
    """

    from elrpy.losses import get_wrapped_loss

    np_dtype = _scipy_dtype(dtype)
    jax_dtype = jnp.dtype(np_dtype.name)
    X, Y, G = group_data
    X = jnp.asarray(X, dtype=jax_dtype)
    Y = jnp.asarray(Y, dtype=jax_dtype)
    G = _as_bcoo(G, jax_dtype)
    theta0 = onp.asarray(jax.device_get(model_params), dtype=onp.float64)

    wrapped_loss = get_wrapped_loss(loss_fn, model_fn)
    value_and_grad = jax.jit(jax.value_and_grad(wrapped_loss))

    def objective(theta):
        loss, grad = value_and_grad(jnp.asarray(theta, dtype=jax_dtype), X, Y, G)
        return float(jax.device_get(loss)), onp.asarray(
            jax.device_get(grad), dtype=onp.float64
        )

    minimize_kwargs: dict[str, Any] = {"jac": True}
    if hessian == "dense":
        hessian_fn = jax.jit(jax.hessian(wrapped_loss))

        def dense_hessian(theta):
            hess = hessian_fn(jnp.asarray(theta, dtype=jax_dtype), X, Y, G)
            return onp.asarray(jax.device_get(hess), dtype=onp.float64)

        minimize_kwargs["hess"] = dense_hessian
    elif hessian == "hvp":
        grad_fn = jax.jit(jax.grad(wrapped_loss))

        @jax.jit
        def hessian_vector_product_jax(theta, vector):
            return jax.jvp(lambda z: grad_fn(z, X, Y, G), (theta,), (vector,))[1]

        def hessian_vector_product(theta, vector):
            hvp = hessian_vector_product_jax(
                jnp.asarray(theta, dtype=jax_dtype),
                jnp.asarray(vector, dtype=jax_dtype),
            )
            return onp.asarray(jax.device_get(hvp), dtype=onp.float64)

        minimize_kwargs["hessp"] = hessian_vector_product
    elif hessian != "none":
        raise ValueError("hessian must be one of 'dense', 'hvp', or 'none'")

    optimizer_options = {"gtol": gtol, "maxiter": maxiter}
    if options is not None:
        optimizer_options.update(options)

    start = perf_counter()
    scipy_result = minimize(
        objective,
        theta0,
        method=method,
        options=optimizer_options,
        **minimize_kwargs,
    )
    seconds = perf_counter() - start
    final_loss, final_grad = objective(scipy_result.x)
    final_grad_norm = float(onp.linalg.norm(final_grad))
    success = bool(scipy_result.success)
    message = str(scipy_result.message)
    accepted_by_loss_stability = False
    if (
        not success
        and onp.dtype(np_dtype) == onp.dtype(onp.float64)
        and 1e-6 <= gtol <= 1e-4
        and final_grad_norm <= 1e-1
        and _no_meaningful_gradient_descent(
            objective,
            onp.asarray(scipy_result.x, dtype=onp.float64),
            final_loss,
            final_grad,
            loss_rtol=1e-5,
        )
    ):
        success = True
        accepted_by_loss_stability = True
        message = f"{message}; accepted as loss-stable"

    return ScipyFitResult(
        params=onp.asarray(scipy_result.x, dtype=onp.float64),
        loss=final_loss,
        grad_norm=final_grad_norm,
        success=success,
        message=message,
        nit=int(getattr(scipy_result, "nit", -1)),
        nfev=int(getattr(scipy_result, "nfev", -1)),
        njev=int(getattr(scipy_result, "njev", -1)),
        nhev=(
            int(getattr(scipy_result, "nhev"))
            if hasattr(scipy_result, "nhev")
            else None
        ),
        seconds=seconds,
        scipy_result=scipy_result,
        stage=stage,
        accepted_by_loss_stability=accepted_by_loss_stability,
    )


def fit_trust_region(
    loss_fn: Callable,
    model_fn: Callable,
    model_params,
    group_data: Tuple[Any, Any, Any],
    *,
    mode: TrustMode = "polished",
    stages: Optional[Sequence[ScipyStage]] = None,
) -> MultiStageFitResult:
    """Run the production staged SciPy optimizer."""

    optimizer_stages = tuple(stages) if stages is not None else default_trust_stages(mode)
    _require_x64_if_needed(optimizer_stages)

    current_params = onp.asarray(jax.device_get(model_params), dtype=onp.float64)
    stage_results: list[ScipyFitResult] = []
    start = perf_counter()
    for stage_index, stage in enumerate(optimizer_stages, start=1):
        print(
            "fit stage "
            f"{stage_index}/{len(optimizer_stages)}: "
            f"method={stage.method} hessian={stage.hessian} "
            f"dtype={stage.dtype} gtol={stage.gtol} maxiter={stage.maxiter}",
            flush=True,
        )
        result = fit_scipy(
            loss_fn,
            model_fn,
            current_params,
            group_data,
            method=stage.method,
            hessian=stage.hessian,
            dtype=stage.dtype,
            gtol=stage.gtol,
            maxiter=stage.maxiter,
            options=stage.options,
            stage=stage,
        )
        stage_results.append(result)
        current_params = result.params
        print(
            "fit stage "
            f"{stage_index}/{len(optimizer_stages)} done: "
            f"success={result.success} loss={result.loss:.12g} "
            f"grad_norm={result.grad_norm:.12g} nit={result.nit} "
            f"nfev={result.nfev} nhev={result.nhev} "
            f"seconds={result.seconds:.3f} message={result.message}",
            flush=True,
        )
    seconds = perf_counter() - start
    final = stage_results[-1]
    return MultiStageFitResult(
        params=final.params,
        loss=final.loss,
        grad_norm=final.grad_norm,
        success=final.success,
        seconds=seconds,
        stages=tuple(stage_results),
    )
