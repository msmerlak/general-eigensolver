"""Anderson-accelerated IPT.

RESULTS.md records Anderson acceleration DIVERGING on SSJ, and this repository
independently confirmed that generator-space momentum fails there too. Both
findings are about SSJ, whose stability comes from a saturation that
extrapolation bypasses -- "every variant that is more faithful to the true
rotation, or skips the reprojection, removes the stabilizer".

IPT is a different map with no such mechanism to break. It is a plain
fixed-point iteration converging linearly at rate rho, which is exactly the
situation Anderson mixing (DIIS, Pulay mixing) was designed for and has been
the workhorse of self-consistent field theory for decades. Two reasons to
expect more than a constant factor:

  * Anderson is a quasi-Newton method on the residual g(x) = T(x) - x, so its
    convergence does not require T to be a contraction -- it can converge
    where the underlying iteration diverges. That attacks IPT's basin, not
    just its rate.
  * Each acceleration step costs a small least-squares over m stored
    residuals: O(N m^2), negligible beside the O(N^2) matvec.

The IPT map here is the column-separable one (a single eigenvector at a time,
diagonal normalization x_j = 1), so the state is a vector and the history is
m vectors.
"""
from __future__ import annotations

import numpy as np

__all__ = ["anderson_ipt_eig"]


def _ipt_step(W, d, x, j):
    """One IPT map application on a single diagonally-normalized column."""
    Wx = W @ x
    lam = d[j] + Wx[j]
    denom = lam - d
    denom[j] = 1.0
    y = Wx / denom
    y[j] = 1.0
    return y, lam


def anderson_ipt_eig(A, target, m=8, beta=1.0, tol=1e-13, max_iter=200,
                     reg=1e-12, return_info=False):
    """One eigenpair by IPT with Anderson mixing.

    Parameters
    ----------
    m : history depth (0 recovers plain IPT).
    beta : mixing parameter; 1.0 is pure Anderson, smaller damps.
    reg : Tikhonov regularization on the least-squares, which matters because
        the residual differences become linearly dependent near convergence.

    Returns (lambda, v) or (..., info) with info["iters"], info["converged"].
    """
    A = np.asarray(A)
    n = A.shape[0]
    d = np.diag(A).astype(np.result_type(A.dtype, np.float64)).copy()
    W = A - np.diag(np.diag(A))
    j = int(target)

    x = np.zeros(n, dtype=np.result_type(d.dtype, np.complex128)
                 if np.iscomplexobj(A) else np.float64)
    x[j] = 1.0
    X_hist, F_hist = [], []
    lam = d[j]
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        gx, lam = _ipt_step(W, d, x, j)
        f = gx - x                                  # residual
        err = float(np.max(np.abs(f)))
        if err <= tol:
            converged = True
            break
        if not np.isfinite(err) or err > 1e12:
            break

        X_hist.append(x.copy())
        F_hist.append(f.copy())
        if len(X_hist) > m + 1:
            X_hist.pop(0)
            F_hist.pop(0)

        k = len(F_hist) - 1
        if m == 0 or k == 0:
            x = x + beta * f                        # plain IPT (beta = 1)
            continue

        # Least-squares over residual DIFFERENCES (the standard, better
        # conditioned form of the constrained problem sum alpha_i = 1).
        dF = np.column_stack([F_hist[i + 1] - F_hist[i] for i in range(k)])
        dX = np.column_stack([X_hist[i + 1] - X_hist[i] for i in range(k)])
        G = dF.conj().T @ dF
        G = G + reg * np.trace(G).real / max(k, 1) * np.eye(k)
        try:
            gamma = np.linalg.solve(G, dF.conj().T @ f)
        except np.linalg.LinAlgError:               # pragma: no cover
            gamma = np.zeros(k, dtype=dF.dtype)
        x = x + beta * f - (dX + beta * dF) @ gamma
        x[j] = 1.0                                  # keep the normalization

    v = x / np.linalg.norm(x)
    if return_info:
        return lam, v, {"iters": it, "converged": converged}
    return lam, v
