"""Spectral divide and conquer for the general eigenproblem, via the matrix
sign function.

Motivation -- the measured inefficiency of the incumbent. On this CPU stack at
N=1000, with one gemm as the unit:

    gemm 1.00 | inverse 5.35 | QR 7.74 | dgeev 93.9 | dgees (Schur) 88.7

A nonsymmetric eigendecomposition is only ~25 N^3 flops, i.e. ~12 gemms' worth
of arithmetic, yet dgeev costs 94 -- Hessenberg reduction and the QR iteration
are bandwidth- and latency-bound and run at roughly an eighth of gemm
efficiency. So a method that performs several times more *arithmetic* still
wins, provided the arithmetic is gemms. That is the opening this module aims
at, and it is a different bet from the rest of the repository: SSJ, IPT and the
shears all descend some invariant by small local updates, and GENERAL.md shows
each of them either stalls or has a bounded basin on non-normal input. Spectral
divide and conquer has no basin at all.

The method (Bai-Demmel-Gu style; the family is classical, nothing here claims
novelty):

  1. sign(A - sigma I) splits the spectrum by which side of the vertical line
     Re(z) = sigma each eigenvalue lies on. P = (I + S)/2 is the spectral
     projector onto the right half, and r = trace(P) is its rank.
  2. An orthonormal basis Q for range(P) (pivoted QR) block-triangularizes:
     Q^T A Q = [[A11, A12], [~0, A22]], with spec(A11) and spec(A22) the two
     halves. This is an ORTHOGONAL similarity, so it is unconditionally
     stable -- unlike the shears of normal.py, nothing here needs a
     well-conditioned eigenbasis.
  3. Recurse on A11 and A22 until blocks are 1x1, or 2x2 holding a complex
     conjugate pair.

The sign iteration itself reuses the repository's own pattern -- an expensive
globally convergent step while far away, a cheap gemm-only step once close:

    scaled Newton   X <- (mu X + mu^-1 X^-1)/2,  mu = |det X|^(-1/n)
    Newton-Schulz   X <- X(3I - X^2)/2                     (2 gemms, no inverse)

Newton is used while ||X^2 - I|| is large (it converges from anywhere with no
eigenvalues on the splitting line) and Newton-Schulz takes over once inside its
region, exactly as QR hands off to Newton-Schulz in ssj.core. Each avoided
inverse saves 5.35 gemm-equivalents and costs 2.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sdc_eigvals", "matrix_sign"]


def matrix_sign(A, tol=1e-12, max_iter=60, ns_switch=0.6, count=None):
    """Matrix sign function by scaled Newton with a Newton-Schulz endgame.

    Returns (S, iters). `count` optionally accumulates a cost model in
    gemm-equivalents (inverse = 5.35, gemm = 1.0) so the benchmark can report
    where the time goes.

    Requires A to have no eigenvalues on the imaginary axis; the caller
    guarantees that by choosing (and if necessary perturbing) the shift.
    """
    n = A.shape[0]
    eye = np.eye(n)
    X = np.asarray(A, dtype=np.float64).copy()
    # Scale into range: the iteration is invariant to positive scaling, and
    # starting near unit norm keeps the first Newton steps well conditioned.
    # The scaling has to be BOTH cheap and tight. np.linalg.norm(X, 2) is an
    # SVD and costs more than several Newton steps; but the cheap bound
    # sqrt(||X||_1 ||X||_inf) overestimates a random matrix badly, which
    # over-scales X, pushes its eigenvalues toward 0 and *adds* Newton steps
    # (measured: 0.23x -> 0.08x of dgeev). Power iteration is both: O(N^2)
    # per step and tight within a few percent.
    nrm = _spectral_norm(X)
    if nrm == 0.0:
        return X, 0
    X /= nrm

    for it in range(1, max_iter + 1):
        X2 = X @ X
        if count is not None:
            count["gemm"] += count["_w"]
        dev = np.linalg.norm(X2 - eye, "fro") / np.sqrt(n)
        if dev < tol:
            return X, it
        if dev < ns_switch:
            # Newton-Schulz: 2 gemms, no factorization, quadratic.
            X = X @ (1.5 * eye - 0.5 * X2)
            if count is not None:
                count["gemm"] += count["_w"]
        else:
            # Scaled Newton with determinantal scaling. One LU serves both
            # the determinant and the inverse -- calling slogdet separately
            # would factorize the same matrix twice.
            Xi, logabsdet = _inv_and_logdet(X)
            if not np.isfinite(logabsdet):
                raise np.linalg.LinAlgError("singular iterate in sign function")
            mu = np.exp(-logabsdet / n)
            if count is not None:
                count["inv"] += count["_w"]
            X = 0.5 * (mu * X + Xi / mu)
    return X, max_iter


def _spectral_norm(X, iters=20):
    """Tight O(N^2) estimate of ||X||_2 by power iteration on X^T X."""
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


def _inv_and_logdet(X):
    """Inverse and log|det| from a single LU factorization."""
    try:
        from scipy.linalg import lu_factor, lu_solve
        lu, piv = lu_factor(X, check_finite=False)
        # A zero pivot means the iterate is singular; log(0) = -inf propagates
        # to a non-finite logabsdet, which the caller turns into a retry with
        # a different shift. Silence the warning rather than the condition.
        with np.errstate(divide="ignore"):
            logabsdet = float(np.sum(np.log(np.abs(np.diag(lu)))))
        Xi = lu_solve((lu, piv), np.eye(X.shape[0]), check_finite=False)
        return Xi, logabsdet
    except ImportError:  # pragma: no cover
        sgn, logabsdet = np.linalg.slogdet(X)
        return np.linalg.inv(X), (logabsdet if sgn != 0 else np.inf)


def _split(A, shift, tol=1e-12, count=None):
    """One spectral split at Re(z) = shift.

    Returns (A11, A22, r) or None when the split is degenerate (everything on
    one side), which tells the caller to try another shift.
    """
    n = A.shape[0]
    if count is not None:
        count["_w"] = (n / count["_N"]) ** 3   # price sub-blocks by size^3
    S, _ = matrix_sign(A - shift * np.eye(n), tol=tol, count=count)
    P = 0.5 * (np.eye(n) + S)
    r = int(np.rint(np.trace(P)))
    if r <= 0 or r >= n:
        return None

    # Orthonormal basis for range(P): pivoted QR puts the r independent
    # columns first. P is a rank-r projector, so its leading r Q-columns span
    # the invariant subspace.
    Q, _, _ = _pivoted_qr(P)
    if count is not None:
        count["qr"] += count["_w"]
    B = Q.T @ (A @ Q)
    if count is not None:
        count["gemm"] += 2 * count["_w"]

    # The (2,1) block is zero in exact arithmetic; how far it misses is the
    # split's backward error and is worth surfacing rather than assuming.
    resid = np.linalg.norm(B[r:, :r], "fro") / max(np.linalg.norm(A, "fro"), 1e-300)
    return B[:r, :r], B[r:, r:], r, resid


def _pivoted_qr(M):
    try:
        from scipy.linalg import qr as _qr
        return _qr(M, mode="economic", pivoting=True)
    except ImportError:  # pragma: no cover
        Q, R = np.linalg.qr(M)
        return Q, R, np.arange(M.shape[1])


def _base_eigvals(A):
    """Eigenvalues of a 1x1 or 2x2 block, in closed form."""
    if A.shape[0] == 1:
        return np.array([complex(A[0, 0])])
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    tr, det = a + d, a * d - b * c
    disc = tr * tr / 4.0 - det
    root = np.sqrt(complex(disc))
    return np.array([tr / 2.0 + root, tr / 2.0 - root])


def sdc_eigvals(A, tol=1e-12, min_block=2, max_depth=64, count=None,
                _depth=0, _rng=None):
    """Eigenvalues of a general real matrix by spectral divide and conquer.

    Globally convergent and free of any near-diagonal or normality
    requirement: every transformation applied is an orthogonal similarity.

    `count` optionally accumulates {"gemm", "inv", "qr"} operation counts.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if _rng is None:
        _rng = np.random.default_rng(0)
    if count is not None and "_N" not in count:
        count["_N"], count["_w"] = n, 1.0
    if n <= min_block or _depth >= max_depth:
        if n <= 2:
            return _base_eigvals(A)
        return np.linalg.eigvals(A).astype(complex)

    # Shift candidates: the mean eigenvalue (= trace/n) splits a spectrum
    # centred anywhere, then progressively perturbed shifts break ties when
    # the spectrum is clustered or symmetric about the first choice.
    centre = float(np.trace(A)) / n
    spread = float(np.linalg.norm(A, "fro")) / np.sqrt(n)
    for attempt in range(12):
        if attempt == 0:
            shift = centre
        else:
            shift = centre + spread * float(_rng.standard_normal()) * 0.5 ** (
                attempt // 4)
        try:
            out = _split(A, shift, tol=tol, count=count)
        except np.linalg.LinAlgError:
            continue
        if out is None:
            continue
        A11, A22, r, resid = out
        if resid > 1e-6:
            continue  # split too inaccurate to trust; try another shift
        left = sdc_eigvals(A11, tol, min_block, max_depth, count,
                           _depth + 1, _rng)
        right = sdc_eigvals(A22, tol, min_block, max_depth, count,
                            _depth + 1, _rng)
        return np.concatenate([left, right])

    # No usable split found (e.g. a genuine multiple eigenvalue filling the
    # block): fall back rather than loop or return something wrong.
    return np.linalg.eigvals(A).astype(complex)
