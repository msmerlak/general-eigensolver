"""Zolotarev's best rational approximation to the sign function.

Newton's sign iteration needs ~log(1/l) steps, where l is the spectrum's
relative distance from the splitting line, because every step doubles the
number of correct digits only once the iterate is already close. Zolotarev
(1877) gives the BEST rational approximation of a given type to sign(x) on
l <= |x| <= 1, and it is good enough that two applications suffice in double
precision (Nakatsukasa & Freund, SIAM Review 2016, "Computing Fundamental
Matrix Decompositions Accurately via the Matrix Sign Function in Two
Iterations: The Power of Zolotarev's Functions").

The type-(2r+1, 2r) approximant is

    Z(x) = M x prod_{j=1..r} (x^2 + c_{2j}) / (x^2 + c_{2j-1})

    c_i = l^2 sn^2(i K / (2r+1); l') / cn^2(i K / (2r+1); l'),
    l' = sqrt(1 - l^2),  K = complete elliptic integral of the first kind,

with M fixed by equioscillation. The matrix version replaces x^2 by A^2 and
each denominator by a linear solve; the r factors are INDEPENDENT, which is
the property the parallel literature is actually buying.

MEASURED BOUNDARY -- read this before reaching for it. Zolotarev is optimal
on a REAL interval l <= |x| <= 1, so it helps exactly when the spectrum lies
there:

    real spectrum (symmetric, N=200):  Newton 26 iters/1.017s
                                       Zolo r=8 3 iters/0.115s   -> 8.8x FASTER
    complex spectrum (Ginibre, N=200): Newton 14 iters/0.155s
                                       Zolo r=8 7 iters/0.256s   -> 1.7x SLOWER

A general nonsymmetric matrix has eigenvalues spread through the complex
plane, where the real-interval optimality does not apply; the iteration still
converges but needs more passes and much more work per pass. This is why the
Nakatsukasa-Freund title reads "the symmetric eigendecomposition and the SVD".
Zolotarev does NOT rescue spectral divide and conquer for the general problem
-- for that, the published direction is Bai-Demmel-Gu's inverse-free iteration
(matrix multiplication and QR, no inversion), a different mechanism.
"""
from __future__ import annotations

import numpy as np

__all__ = ["zolotarev_coeffs", "zolotarev_sign_scalar", "matrix_sign_zolo"]


def zolotarev_coeffs(ell, r):
    """Coefficients c_1..c_2r and the normalization M for Z of type (2r+1, 2r)
    approximating sign(x) on ell <= |x| <= 1.

    Returns (c, M) with c the 2r coefficients (1-indexed in the formula, plain
    0-indexed array here).
    """
    from scipy.special import ellipj, ellipk

    if not (0.0 < ell < 1.0):
        raise ValueError("ell must lie in (0, 1)")
    m = 1.0 - ell * ell            # scipy parametrizes by m = k^2, k = l'
    K = ellipk(m)
    i = np.arange(1, 2 * r + 1)
    sn, cn, _, _ = ellipj(i * K / (2 * r + 1), m)
    c = ell * ell * sn * sn / (cn * cn)

    # Normalization: with M = 1 the product maps [ell, 1] onto some [lo, hi];
    # the equioscillating choice recentres that onto 1.
    x = np.geomspace(ell, 1.0, 400)
    z = _apply_scalar(x, c, 1.0)
    M = 2.0 / (z.min() + z.max())
    return c, M


def partial_fractions(c):
    """Residues a_j for the partial-fraction form of the Zolotarev product.

    prod_j (t + c_2j)/(t + c_2j-1) = 1 + sum_j a_j/(t + c_2j-1),  t = x^2

    The product form must NOT be applied factor-by-factor to a matrix: the
    individual factors have wildly different scales and only cancel at the
    end, so the intermediate iterates overflow (measured: ||Y|| reaching 1e18
    over r = 8 factors on a well-behaved symmetric matrix, ending in a
    guaranteed overflow). The partial-fraction form has no such intermediates,
    and its r terms are mutually independent -- which is also exactly the
    parallelism the literature is buying.
    """
    r = len(c) // 2
    poles = c[0::2]        # c_1, c_3, ... (denominator constants)
    zeros = c[1::2]        # c_2, c_4, ... (numerator constants)
    a = np.empty(r)
    for j in range(r):
        num = np.prod(zeros - poles[j])
        den = np.prod([poles[i] - poles[j] for i in range(r) if i != j])
        a[j] = num / den
    return poles, a


def _apply_scalar(x, c, M):
    x = np.asarray(x, dtype=float)
    x2 = x * x
    poles, a = partial_fractions(c)
    acc = np.ones_like(x)
    for j in range(len(poles)):
        acc = acc + a[j] / (x2 + poles[j])
    return M * x * acc


def zolotarev_sign_scalar(x, ell, r):
    """Z(x) evaluated on scalars/arrays -- used to validate the coefficients
    independently of any matrix machinery."""
    c, M = zolotarev_coeffs(ell, r)
    return _apply_scalar(np.asarray(x, dtype=float), c, M)


def _cond_estimate(A):
    """Cheap lower bound on the spectrum's relative distance to the splitting
    line: sigma_min / sigma_max, estimated from an LU condition estimator.

    Underestimating is safe -- it only raises r, costing work but not
    correctness.
    """
    try:
        from scipy.linalg import lu_factor
        from scipy.linalg.lapack import dgecon
        lu, piv = lu_factor(A, check_finite=False)
        anorm = float(np.abs(A).sum(axis=0).max())   # 1-norm
        rcond, _ = dgecon(lu, anorm)
        return max(float(rcond), 1e-16)
    except Exception:
        return 1e-8


def matrix_sign_zolo(A, tol=1e-12, r=8, max_iter=8, count=None, ell=None):
    """Matrix sign function via repeated Zolotarev application.

    Returns (S, iterations). Each iteration costs one gemm for A^2 plus r
    solves and r gemms; two iterations normally suffice in double precision,
    against Newton's ~12.
    """
    n = A.shape[0]
    eye = np.eye(n)
    X = np.asarray(A, dtype=np.float64).copy()
    nrm = _spectral_norm(X)
    if nrm == 0.0:
        return X, 0
    X /= nrm

    if ell is None:
        ell = _cond_estimate(X)
    ell = min(max(ell, 1e-15), 0.99)

    w = 1.0 if count is None else count.get("_w", 1.0)
    for it in range(1, max_iter + 1):
        X2 = X @ X
        if count is not None:
            count["gemm"] += w
        dev = np.linalg.norm(X2 - eye, "fro") / np.sqrt(n)
        if dev < tol:
            return X, it

        c, M = zolotarev_coeffs(ell, r)
        poles, a = partial_fractions(c)
        acc = eye.copy()
        for j in range(r):
            acc = acc + a[j] * np.linalg.inv(X2 + poles[j] * eye)
            if count is not None:
                count["solve"] = count.get("solve", 0.0) + w
        X = M * (X @ acc)
        if count is not None:
            count["gemm"] += w
        if not np.all(np.isfinite(X)):
            raise np.linalg.LinAlgError(
                "Zolotarev iterate overflowed: the spectrum is not confined to "
                "the real interval the approximation is optimal on")
        # After one pass the spectrum is squeezed toward +-1, so the next pass
        # needs a far milder ell. Frobenius, not the 2-norm: an SVD here would
        # cost more than the pass it is measuring.
        d = float(np.linalg.norm(X @ X - eye, "fro")) / np.sqrt(n)
        ell = min(max(1.0 - 2.0 * d, 1e-3), 0.99)
    return X, max_iter


def _spectral_norm(X, iters=20):
    n = X.shape[0]
    v = np.ones(n) + 0.01 * np.arange(n)
    v /= np.linalg.norm(v)
    est = 0.0
    for _ in range(iters):
        w = X.T @ (X @ v)
        est = float(np.linalg.norm(w))
        if est == 0.0:
            return 0.0
        v = w / est
    return float(np.sqrt(est))
