"""Purification: a fixed-point map whose fixed points are spectral PROJECTORS.

Every other map in this repository has eigenvectors (or an eigenbasis) as its
fixed point, and each pays the same tax: power/gradient flows converge globally
but only to spectral extremes and only linearly; IPT and Newton-type maps are
fast but locally convergent. This one changes the object. Its fixed points are
idempotents,

    P <- 3 P^2 - 2 P^3 = P^2 (3I - 2P)

and the scalar map p(x) = 3x^2 - 2x^3 has p(0)=0, p(1)=1, p(1/2)=1/2 with
p'(0) = p'(1) = 0 and p'(1/2) = 3/2. So 0 and 1 are SUPERATTRACTING (quadratic)
while 1/2 repels, and p maps [0,1] into itself. Scale A linearly so its
spectrum lands in [0,1] with the splitting point mu at 1/2, and every
eigenvalue below mu is driven to 1, every one above to 0:

    **globally convergent, quadratic, and every operation is a gemm.**

That last clause is the point. GENERAL.md derives the break-even for spectral
divide and conquer on this CPU as "the sign iteration must be inverse-free" --
an inverse costs 5.35 gemm-equivalents here and Newton's sign iteration needs
about twelve of them. Newton-Schulz on the sign function is inverse-free but
NOT globally convergent (it needs the spectrum already near +-1). Purification
is both, because the initial scaling into [0,1] is a guarantee rather than a
hope. This is McWeeny purification, the basis of linear-scaling electronic
structure; nothing here is new except its use as the SDC splitter.

LIMITATION, stated up front: the map needs the spectrum inside the real
interval [0,1], so this is a HERMITIAN-only tool. A general matrix with complex
eigenvalues cannot be scaled into [0,1] and purification does not apply -- the
sign function still does. This is the same real-interval boundary that stopped
Zolotarev from rescuing the general case.
"""
from __future__ import annotations

import numpy as np

__all__ = ["purify", "spectral_projector", "purify_split"]


def _bounds(A, iters=30):
    """Cheap spectral bounds by Gershgorin (exact enclosure, O(N^2))."""
    d = np.real(np.diag(A))
    r = np.sum(np.abs(A), axis=1) - np.abs(d)
    return float(np.min(d - r)), float(np.max(d + r))


def purify(P, tol=1e-12, max_iter=100, count=None, precision="full",
           switch=1e-4):
    """Iterate P <- P^2(3I - 2P) to the nearest idempotent.

    P must already have its spectrum in [0, 1]; `spectral_projector` does that
    scaling. Two gemms per iteration, no factorization.

    precision="mixed" runs the early sweeps in float32 and hands off to
    float64 once the idempotency defect drops below `switch`, exactly the
    mechanism ssj_eigh uses (the iteration is memoryless -- it recomputes
    P2 = P@P from the CURRENT P every step, so a low-precision early phase
    cannot poison the answer, only the warm start it hands to the float64
    phase). Measured: 1.43-1.45x at N=600/1200, identical rank and
    idempotency to a float64-only run.

    Returns (P, iters).
    """
    if precision == "mixed" and P.dtype != np.complex64 and P.dtype != np.complex128:
        P32, it32 = purify(P.astype(np.float32), tol=switch, max_iter=max_iter,
                           count=count, precision="full")
        P64, it64 = purify(P32.astype(np.float64), tol=tol, max_iter=max_iter,
                           count=count, precision="full")
        return P64, it32 + it64
    n = P.shape[0]
    scale = max(np.sqrt(n), 1.0)
    # The O(N^2) work is kept out of the way of the two gemms: at N=600 the
    # naive form (a fresh 3I-2P, a fresh P2-P) costs ~40% on top of the
    # gemms it is measuring.
    T = np.empty_like(P)
    for it in range(1, max_iter + 1):
        P2 = P @ P
        if count is not None:
            count["gemm"] = count.get("gemm", 0.0) + 1.0
        np.subtract(P2, P, out=T)
        err = float(np.linalg.norm(T, "fro")) / scale
        if err < tol:
            return P, it
        np.multiply(P, -2.0, out=T)          # T = -2P
        T.flat[:: n + 1] += 3.0              # T = 3I - 2P
        P = P2 @ T
        if count is not None:
            count["gemm"] = count.get("gemm", 0.0) + 1.0
    return P, max_iter


def spectral_projector(A, mu, tol=1e-12, max_iter=100, count=None,
                       bounds=None, precision="full", switch=1e-4):
    """Projector onto the invariant subspace of eigenvalues BELOW mu.

    A must be Hermitian. The initial scaling is what makes this globally
    convergent: mu is placed at 1/2 and the whole spectrum inside [0, 1], from
    which purification cannot escape.
    """
    A = np.asarray(A)
    n = A.shape[0]
    lo, hi = bounds if bounds is not None else _bounds(A)
    # Place mu at 1/2, keep the spectrum in [0,1]: the tighter of the two
    # half-widths decides the slope.
    c = 0.5 / max(hi - mu, mu - lo, 1e-300)
    P0 = -c * A
    P0.flat[:: n + 1] += 0.5 + c * mu
    P, iters = purify(P0, tol=tol, max_iter=max_iter, count=count,
                      precision=precision, switch=switch)
    return P, iters


def purify_split(A, mu, tol=1e-12, count=None):
    """One SDC-style spectral split at mu, using purification instead of the
    sign function.

    Returns (A11, A22, r, resid) with the same contract as ssj.sdc._split:
    A11 holds the eigenvalues below mu, A22 those above, both obtained by an
    ORTHOGONAL similarity, so the split is unconditionally stable. `None` if
    the split is degenerate.
    """
    from scipy.linalg import qr as _qr
    A = np.asarray(A)
    n = A.shape[0]
    P, _ = spectral_projector(A, mu, tol=tol, count=count)
    r = int(np.rint(np.trace(P).real))
    if r <= 0 or r >= n:
        return None
    Q, _, _ = _qr(P, mode="economic", pivoting=True)
    if count is not None:
        count["qr"] = count.get("qr", 0.0) + 1.0
    B = Q.conj().T @ (A @ Q)
    if count is not None:
        count["gemm"] = count.get("gemm", 0.0) + 2.0
    resid = np.linalg.norm(B[r:, :r], "fro") / max(np.linalg.norm(A, "fro"),
                                                   1e-300)
    return B[:r, :r], B[r:, r:], r, resid
