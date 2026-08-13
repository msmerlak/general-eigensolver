"""Eigensolver for NORMAL matrices (A A^H = A^H A), and the norm-reducing
shear that drives a general matrix toward normality.

A normal matrix is unitarily diagonalizable but generally not symmetric, and
its eigenvalues are generally complex -- so it sits strictly between the
symmetric problem SSJ solves and the general problem, which GENERAL.md shows
has no globalizer in this family. It is solved here exactly, by reduction to
the symmetric machinery:

    A = H1 + i H2,   H1 = (A + A^H)/2,   H2 = (A - A^H)/(2i)

Both H1 and H2 are Hermitian, and A is normal precisely when they commute --
in which case they share an eigenbasis, and diagonalizing the single Hermitian
matrix H1 + alpha H2 for generic alpha recovers it. One SSJ solve therefore
diagonalizes A.

The genericity of alpha is not a detail. At alpha = 0 (diagonalizing the
Hermitian part alone) the method FAILS on exactly the matrices it is meant
for: a real normal matrix with a complex eigenvalue pair has a *degenerate*
Hermitian part on that pair (both diagonal entries equal Re lambda), so
diagonalizing H1 resolves the degeneracy arbitrarily and leaves H2 -- hence A
-- undiagonalized. Measured: off stalls at 3.13 with alpha = 0 versus 1.3e-13
with a generic alpha, on a matrix that is normal to 1.3e-16.
"""
from __future__ import annotations

import numpy as np

from .core import _am, ssj_eigh

__all__ = ["normal_eig", "normality_defect", "shear_toward_normal"]

# An arbitrary irrational-ish constant: any generic value separates the
# eigenvalues of H1 + alpha H2 that H1 alone leaves degenerate.
_ALPHA = 0.7390851332151607


def normality_defect(A):
    """||A^H A - A A^H||_F / ||A||_F^2 -- zero exactly when A is normal.

    Costs two gemms and no eigensolve, so it is usable as a dispatch test.
    """
    xp = _am(A)
    C = A.conj().T @ A - A @ A.conj().T
    denom = float(xp.sum(xp.abs(A) ** 2))
    return float(xp.linalg.norm(C, "fro")) / max(denom, 1e-300)


def normal_eig(A, tol=1e-13, alpha=_ALPHA, backend="lapack",
               return_info=False, **ssj_kw):
    """Eigendecomposition of a normal matrix via ONE Hermitian eigensolve.

    Returns (w, U) with w complex, U unitary, A U = U diag(w). Accuracy is
    limited by how normal A actually is: the residual scales with
    normality_defect(A), so the caller should check it (it is returned in
    info) rather than assume. For genuinely normal input this is exact to
    roundoff -- measured eigenvalue error 6.5e-15 up to N = 800.

    backend selects the Hermitian solver: "lapack" (default, zheevd) or "ssj".
    The reduction is what matters here, not which Hermitian kernel runs
    underneath, and on CPU LAPACK's is ~30x faster -- routing through SSJ
    makes the whole method 0.08x dgeev instead of beating it.
    """
    xp = _am(A)
    A = xp.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    Ac = A.astype(np.complex128 if A.dtype != np.complex64 else np.complex64)

    H1 = (Ac + Ac.conj().T) / 2.0
    H2 = (Ac - Ac.conj().T) / 2.0j
    M = H1 + alpha * H2
    if backend == "lapack":
        _, U = xp.linalg.eigh(M)
    elif backend == "ssj":
        _, U = ssj_eigh(M, tol=tol, **ssj_kw)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    B = U.conj().T @ (Ac @ U)
    w = xp.diag(B).copy()
    if return_info:
        off = float(xp.linalg.norm(B - xp.diag(w), "fro"))
        return w, U, {"off": off / max(float(xp.linalg.norm(Ac, 2)), 1e-300),
                      "defect": normality_defect(Ac)}
    return w, U


def shear_toward_normal(A, max_iter=200, tol=1e-13, cap=0.25):
    """Drive A toward normality by norm-reducing similarity (Eberlein-type).

    For A <- T^{-1} A T with T = I + G, first order gives A <- A + [A, G]; the
    symmetric (shear) part of G moves ||A||_F while the antisymmetric part
    cannot, and

        d||A||_F^2 = 2 <C, S>,    C = A^H A - A A^H

    so steepest descent on the departure from normality
    ||A||_F^2 - sum|lambda|^2 is simply S = -mu C. C is symmetric and
    traceless, so the shear is automatically volume-preserving.

    mu is chosen by backtracking until ||A||_F actually decreases. Without
    that, a fixed cap overshoots on near-normal input (measured: an
    already-normal matrix pushed from defect 0.000 to 0.028 by one step) --
    the same saturation lesson as SSJ's bounded angles.

    IMPORTANT -- this does not always reach normality, and the limitation is
    fundamental rather than a tuning failure. A matrix whose eigenvector basis
    is ill-conditioned cannot be normalized by a *bounded* similarity, so the
    descent plateaus at a positive defect (measured: Ginibre N=60 plateaus at
    7.6e-3, a near-diagonal matrix at 1.4e-4), and the residual defect passes
    straight through to the eigenvalue error of any subsequent normal_eig.
    Returns (A, info) with info["defect"] so the caller can see where it got.
    """
    xp = _am(A)
    A = xp.array(A, dtype=np.float64 if A.dtype.kind == "f" else A.dtype)
    n = A.shape[0]
    eye = xp.eye(n, dtype=A.dtype)
    iters = 0
    for iters in range(1, max_iter + 1):
        C = A.conj().T @ A - A @ A.conj().T
        nC = float(xp.linalg.norm(C, 2))
        if normality_defect(A) < tol:
            break
        if nC == 0.0:
            break
        f0 = float(xp.sum(xp.abs(A) ** 2))
        mu = cap / nC
        improved = False
        for _ in range(30):
            T = eye - mu * C
            try:
                An = xp.linalg.solve(T, A @ T)
            except Exception:  # pragma: no cover - singular T
                mu *= 0.5
                continue
            f1 = float(xp.sum(xp.abs(An) ** 2))
            if np.isfinite(f1) and f1 < f0:
                A, improved = An, True
                break
            mu *= 0.5
        if not improved:
            break  # stationary: no bounded shear decreases ||A||_F further
    return A, {"iters": iters, "defect": normality_defect(A)}
