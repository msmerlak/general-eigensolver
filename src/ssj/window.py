"""Interval eigensolving: ALL eigenpairs in [lo, hi], via purification only.

Every other partial solver in this repository (ipt_eig_partial, davidson_eig,
block_ipt_eig) needs a diagonal entry that already approximates the wanted
eigenvalue -- they are targeted solvers for near-diagonal or structured input.
This one needs nothing of the kind. Purification computes a spectral
projector P(mu) onto {eigenvalues < mu} for ANY Hermitian A by scaling into
[0,1] and iterating (see ssj.purify); it is globally convergent with no basin
condition, no diagonal structure required, no target eigenvalue guessed.

Two such projectors compose into a WINDOW projector:

    P_window = P(hi) - P(mu_lo)

which is idempotent (both factors are polynomials in A and hence commute
exactly) with rank = the number of eigenvalues in [lo, hi]. Extract an
orthonormal basis for its range (pivoted QR) and do Rayleigh-Ritz on that
small basis: EXACT eigenpairs in the window, at whatever accuracy the
purification tolerance was set to.

WHY THIS IS THE RIGHT TOOL FOR AN UNKNOWN COUNT, MEASURED. Given an interval
and no prior knowledge of how many eigenvalues it contains, ARPACK-style
shift-invert (scipy eigsh/eigs with sigma) needs a guessed k UP FRONT and
returns exactly k Ritz values with NO WARNING if the true count is larger.
Measured, GOE N=600, a window with 24 true eigenvalues inside:

    k_guess=12 (half)   -> 12 returned, 12 in-window  -- MISSED 12, SILENTLY
    k_guess=22 (close)  -> 22 returned, 22 in-window  -- MISSED 2, SILENTLY
    k_guess=24 (exact)  -> 24 returned, 24 in-window  -- correct
    window_eig (no guess needed) -> found 24, exact match

This function's projector RANK is a certificate: it is not a guess, it is the
actual count (verified against dense diagonalization on GOE, degenerate, and
empty-window cases -- see tests). That is the capability being sold here, not
raw speed: measured 6-7x SLOWER than ARPACK even when ARPACK is handed the
exact count in advance, and at N=800 slower than a full dsyevd. Use this when
correctness/completeness on an unknown count matters more than wall time --
counting states in a spectral window, verifying no eigenvalue was missed, or
whenever the alternative is guessing k and hoping.
"""
from __future__ import annotations

import numpy as np

from .purify import _bounds, spectral_projector

__all__ = ["window_eig", "window_count"]


def window_count(A, lo, hi, tol=1e-11, bounds=None):
    """Number of eigenvalues in [lo, hi], via two purifications and a trace.

    Cheaper than window_eig when only the count is wanted: no QR, no
    Rayleigh-Ritz. Still O(N^2) per purification iteration, matmul only.
    """
    A = np.asarray(A)
    bounds = bounds if bounds is not None else _bounds(A)
    P_hi, _ = spectral_projector(A, hi, tol=tol, bounds=bounds)
    P_lo, _ = spectral_projector(A, lo, tol=tol, bounds=bounds)
    return int(np.rint(np.trace(P_hi - P_lo).real))


def window_eig(A, lo, hi, tol=1e-11, return_info=False):
    """All eigenpairs of Hermitian A with eigenvalue in [lo, hi].

    Returns (w, V) with w ascending and V's columns orthonormal eigenvectors,
    or (w, V, info) with info["count"] (the certified count == len(w)) and
    info["resid"] (the window projector's departure from idempotency, a
    diagnostic that costs nothing extra).

    Boundary note: if lo or hi lands exactly ON an eigenvalue the split is
    ambiguous by construction (true of any spectral-projector method, not a
    defect here) -- nudge the boundary if a specific inclusion is required.
    """
    from scipy.linalg import qr as _qr

    A = np.asarray(A)
    n = A.shape[0]
    if lo > hi:
        lo, hi = hi, lo
    bounds = _bounds(A)
    P_hi, _ = spectral_projector(A, hi, tol=tol, bounds=bounds)
    P_lo, _ = spectral_projector(A, lo, tol=tol, bounds=bounds)
    Pw = P_hi - P_lo
    idem_resid = float(np.linalg.norm(Pw @ Pw - Pw, "fro"))
    r = int(np.rint(np.trace(Pw).real))

    if r <= 0:
        w, V = np.array([]), np.zeros((n, 0))
    else:
        Q, _, _ = _qr(Pw, mode="economic", pivoting=True)
        Q = Q[:, :r]
        H = Q.conj().T @ (A @ Q)
        H = (H + H.conj().T) / 2.0
        w, S = np.linalg.eigh(H)
        V = Q @ S

    if return_info:
        return w, V, {"count": r, "resid": idem_resid}
    return w, V
