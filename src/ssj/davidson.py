"""Davidson: the rewriting that actually beat IPT.

Surveying rewritings of A v = lambda v as fixed points (experiments_zoo.py),
most changed nothing. This one changes the basin, and the reason is structural:
it is the only variant tried that does not commit to a single iterate.

IPT applies the diagonal resolvent to the whole vector and replaces v outright.
Davidson applies the same diagonal resolvent to the RESIDUAL only,

    t = (lambda I - D)^{-1} (A u - lambda u)

and then, instead of stepping to t, appends it to a subspace and takes the best
vector the subspace can offer by Rayleigh-Ritz. The preconditioner is
identical -- the difference is entirely that the correction accumulates rather
than replaces. Near-degenerate levels, which defeat IPT because their terms
dominate the perturbation sum, are simply resolved by the small Rayleigh-Ritz
eigenproblem instead. That is the same medicine as block IPT, but the block is
built automatically out of whatever directions the residual actually explores.

Measured (symmetric near-diagonal, N=300, target mid-spectrum, relative
residual 1e-13):

    coupling   IPT        Davidson
    0.5        21 its     14 its       (faster where both work)
    2          DIVERGES   29 its
    8          DIVERGES   96 its
    30         diverges   diverges     (eigenvalue still good to 2.6e-5)

A basin ~16x wider in coupling than plain IPT, and quicker inside it.

Two implementation points that mattered, both found by getting them wrong:
reorthogonalize the correction TWICE (one Gram-Schmidt pass loses orthogonality
as the subspace grows), and use a RELATIVE residual test -- with ||A|| ~ 300 an
absolute 1e-12 threshold reports divergence on a run that has actually reached
machine precision (measured eigenvalue error 9.4e-17 on a "failed" run).
"""
from __future__ import annotations

import numpy as np

__all__ = ["davidson_eig"]


def davidson_eig(A, target, rtol=1e-13, max_iter=400, max_subspace=150,
                 return_info=False):
    """One eigenpair by Davidson's method, targeted by a diagonal entry.

    `target` selects the eigenvalue nearest A[target, target], exactly as in
    IPT, so this is a drop-in for the same use case with a wider basin.
    Hermitian A. Returns (lambda, v) or (..., info).
    """
    A = np.asarray(A)
    n = A.shape[0]
    d = np.real(np.diag(A)).copy()
    j = int(target)
    scale = float(np.linalg.norm(A, 2)) or 1.0

    V = np.zeros((n, 1))
    V[j, 0] = 1.0
    lam, u = d[j], V[:, 0]
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        H = V.T @ (A @ V)
        H = (H + H.T) / 2.0
        w, S = np.linalg.eigh(H)
        m = int(np.argmin(np.abs(w - d[j])))
        lam = float(w[m])
        u = V @ S[:, m]
        r = A @ u - lam * u
        if np.linalg.norm(r) / scale <= rtol:
            converged = True
            break

        den = lam - d
        den[np.abs(den) < 1e-12] = 1e-12
        t = r / den                       # the SAME preconditioner IPT uses
        for _ in range(2):                # twice: one pass is not enough
            t = t - V @ (V.T @ t)
        nt = np.linalg.norm(t)
        if nt < 1e-14:
            break
        if V.shape[1] >= max_subspace:    # restart on the current Ritz vector
            V = (u / np.linalg.norm(u))[:, None]
            t = r / den
            for _ in range(2):
                t = t - V @ (V.T @ t)
            nt = np.linalg.norm(t)
            if nt < 1e-14:
                break
        V = np.column_stack([V, t / nt])

    u = u / np.linalg.norm(u)
    if return_info:
        return lam, u, {"iters": it, "converged": converged,
                        "subspace": V.shape[1]}
    return lam, u
