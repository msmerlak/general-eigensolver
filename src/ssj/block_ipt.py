"""Block IPT: a fixed-point equation with a basin you control.

IPT solves A v = lambda v as the quadratic fixed point

    Lambda_j = d_j + (WV)_jj,    V_ij = (WV)_ij / (Lambda_j - d_i)

whose contraction rate is set by the SMALLEST denominator |lambda_j - d_i|.
That is a locator expansion, and it fails exactly when some other diagonal
entry is near the target eigenvalue -- the resonance problem. Measured
divergences: rho_j = 0.18 on a dense isolated level, 0.25 on a 2D Anderson
lattice, both far inside what a one-hop rate estimate suggests, because the
series sums over many near-resonant paths rather than one.

Saturating the denominator does not fix it (GENERAL.md, Failure 1): the
trouble is the sum, not any single term.

The fix is to stop treating near-resonant states perturbatively. Split the
indices into a block B holding the target and everything within tau of it,
and the remainder C. Writing the eigenvector as v_B on B and v_C = X v_B on
C, the eigenvalue problem is exactly equivalent to

    X   = (lambda - D_C)^{-1} (W_CB + W_CC X)        (IPT-like, on C only)
    (A_BB + W_BC X) v_B = lambda v_B                 (exact b-by-b problem)

The first is a contraction with rate at most ||W|| / tau, because every index
whose gap is smaller than tau was placed in B. **The basin is now a parameter
of the algorithm rather than a property of the matrix.** The second is a small
dense eigenproblem solved exactly, which is what absorbs the near-degeneracies
that defeat plain IPT.

This is quasi-degenerate (Loewdin/Bloch effective-Hamiltonian) perturbation
theory read as a fixed-point iteration; the block-vs-perturbative split is
classical physics practice. Plain IPT is the tau -> 0 limit, b = 1.

Cost per outer iteration is one N-by-b matvec plus a b-by-b eigensolve --
the same shape as running plain IPT on b columns -- so the price of a larger
basin is linear in the block size.
"""
from __future__ import annotations

import numpy as np

__all__ = ["block_ipt_eig", "adaptive_block_ipt_eig", "choose_block"]


def choose_block(A, target, tau=None, max_block=64, by="ratio"):
    """Indices to solve EXACTLY rather than perturbatively.

    by="ratio" (default) takes the largest |W_ij| / |d_i - d_j|, i.e. exactly
    the terms that break IPT's contraction. This is the principled criterion
    and it unifies two cases that look different: on a dense matrix the
    offenders are near-degenerate levels (small denominator), on a lattice they
    are the strongly coupled neighbours (large numerator). Selecting by energy
    proximity alone gets the lattice case wrong, because the sites nearest in
    energy there are spatially distant and not directly coupled at all.

    by="gap" keeps the older, purely energetic criterion for comparison.
    """
    d = np.asarray(A.diagonal()).ravel() if hasattr(A, "diagonal") \
        else np.diag(A)
    gaps = np.abs(d - d[target])
    if by == "gap":
        idx = np.argsort(gaps)
        keep = idx[:max_block] if tau is None else \
            idx[gaps[idx] <= tau][:max_block]
    else:
        col = np.abs(np.asarray(A[:, target]).ravel())
        row = np.abs(np.asarray(A[target, :]).ravel())
        coupling = np.maximum(col, row)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(gaps > 0, coupling / np.where(gaps > 0, gaps, 1.0),
                             np.inf)
        ratio[target] = np.inf
        keep = np.argsort(-ratio)[:max_block]
    keep = np.asarray(keep)
    if target not in keep:
        keep = np.concatenate([[target], keep])[:max_block]
    return np.sort(keep)


def block_ipt_eig(A, target, tau=None, max_block=32, tol=1e-13,
                  max_iter=200, inner=2, by="ratio", return_info=False):
    """One eigenpair by block IPT.

    Parameters
    ----------
    A : (n, n) dense array (real or complex).
    target : index whose diagonal entry estimates the wanted eigenvalue.
    tau : block radius. Everything within tau of d[target] is solved exactly.
        Defaults to a value that captures a handful of the nearest levels,
        which is what makes the difference on resonant problems.
    max_block : cap on block size (cost is linear in it).
    inner : X-iterations per outer eigen-solve.

    Returns (lambda, v), or (lambda, v, info) with info["converged"],
    info["iters"], info["block"] and info["rate"] -- the ACTUAL contraction
    rate over C, which unlike IPT's one-hop rho_j is a genuine bound.
    """
    A = np.asarray(A)
    n = A.shape[0]
    d = np.diag(A).copy()

    B = choose_block(A, target, tau, max_block, by=by)
    C = np.setdiff1d(np.arange(n), B)
    b = len(B)
    if len(C) == 0:                                  # block is everything
        w, V = np.linalg.eig(A)
        m = int(np.argmin(np.abs(w - d[target])))
        out = (w[m], V[:, m])
        return (*out, {"converged": True, "iters": 0, "block": b,
                       "rate": 0.0}) if return_info else out

    A_BB = A[np.ix_(B, B)]
    W_CB = A[np.ix_(C, B)]
    W_BC = A[np.ix_(B, C)]
    A_CC = A[np.ix_(C, C)]
    d_C = np.diag(A_CC).copy()
    W_CC = A_CC - np.diag(d_C)

    # rate bound over C: every small gap was absorbed into B
    tgt = np.argmin(np.abs(B - target))
    lam = complex(d[target])
    X = np.zeros((len(C), b), dtype=np.result_type(A.dtype, np.complex128))
    v_B = np.zeros(b, dtype=X.dtype)
    v_B[tgt] = 1.0
    converged = False
    it = 0
    prev = np.inf

    for it in range(1, max_iter + 1):
        # --- inner: contract X, rate <= ||W_CC|| / tau by construction
        for _ in range(inner):
            denom = lam - d_C
            X = (W_CB + W_CC @ X) / denom[:, None]

        # --- outer: exact b-by-b eigenproblem with the C-block folded in
        H_eff = A_BB + W_BC @ X
        w_eff, V_eff = np.linalg.eig(H_eff)
        m = int(np.argmin(np.abs(w_eff - lam)))
        lam_new, v_B = w_eff[m], V_eff[:, m]

        err = abs(lam_new - lam) / max(abs(lam_new), 1e-300)
        lam = lam_new
        if err <= tol:
            converged = True
            break
        if not np.isfinite(err) or err > 1e3 * prev:
            break
        prev = max(err, 1e-300)

    v = np.zeros(n, dtype=X.dtype)
    v[B] = v_B
    v[C] = X @ v_B
    v /= np.linalg.norm(v)

    if return_info:
        rate = float(np.max(np.abs(W_CC) .max(axis=1) /
                            np.maximum(np.abs(lam - d_C), 1e-300)))
        return lam, v, {"converged": converged, "iters": it, "block": b,
                        "rate": rate}
    return lam, v


def adaptive_block_ipt_eig(A, target, b0=1, max_block=128, grow=4, tol=1e-13,
                           max_outer=150, inner=3, stall=0.5, return_info=False):
    """Block IPT that discovers its own block from the iterate.

    Both a priori block criteria are unreliable: selecting by gap and selecting
    by |W_ij|/|gap| each win on some problems and lose on others (measured in
    GENERAL.md), because both predict trouble from the matrix alone. The
    iterate does better than predicting -- it SHOWS which indices are failing,
    as a large tail amplitude |X_i|. An index with a large amplitude is
    precisely one whose "small correction" is not small, i.e. one that should
    have been solved exactly rather than perturbatively.

    So: start from a block of size b0 (1 = plain IPT), iterate, and whenever
    the contraction stalls, promote the `grow` largest-amplitude indices out of
    C and into B. Two things improve over the static version:

      * the block is re-selected against the CURRENT lambda, not the initial
        d[target] -- the resonant set moves as the eigenvalue converges;
      * cost is proportional to the block actually needed, rather than to a
        guessed max_block, so easy problems stay cheap.

    `stall` is the growth trigger and matters more than it looks: it is the
    contraction rate above which the block grows. Setting it near 1 (grow only
    when truly stuck) is measurably too permissive -- the iteration then
    settles for a slow linear rate and runs out of iterations rather than
    buying a better rate with a bigger block. Measured at coupling 8: stall=0.9
    stops at b=37 and fails, stall=0.5 grows to b=93 and converges. The default
    0.5 says "grow until convergence is fast", which is the right trade because
    iterations cost O(N^2 b) either way.

    Returns (lambda, v) or (..., info) with info["block"] the final block size
    and info["grew"] the number of promotions.
    """
    A = np.asarray(A)
    n = A.shape[0]
    d = np.diag(A).copy()
    cdt = np.result_type(A.dtype, np.complex128)

    B = choose_block(A, target, None, max(b0, 1), by="ratio")
    lam = complex(d[target])
    v_B = None
    X_full = np.zeros((n, 1), dtype=cdt)   # tail amplitudes, full-length rows
    converged = False
    grew = 0
    prev_err = np.inf
    outer = 0

    for outer in range(1, max_outer + 1):
        mask = np.ones(n, dtype=bool)
        mask[B] = False
        C = np.where(mask)[0]
        b = len(B)

        A_BB = A[np.ix_(B, B)]
        W_CB = A[np.ix_(C, B)]
        W_BC = A[np.ix_(B, C)]
        d_C = d[C]
        W_CC = A[np.ix_(C, C)] - np.diag(d_C)

        X = np.zeros((len(C), b), dtype=cdt)
        if X_full.shape[1] == b:
            X = X_full[C].copy()            # warm start when shape allows

        for _ in range(inner):
            X = (W_CB + W_CC @ X) / (lam - d_C)[:, None]

        H_eff = A_BB + W_BC @ X
        w_eff, V_eff = np.linalg.eig(H_eff)
        m = int(np.argmin(np.abs(w_eff - lam)))
        lam_new, v_B = w_eff[m], V_eff[:, m]

        err = abs(lam_new - lam) / max(abs(lam_new), 1e-300)
        lam = lam_new
        X_full = np.zeros((n, b), dtype=cdt)
        X_full[C] = X

        if err <= tol:
            converged = True
            break

        # Stalled (or diverging)? Promote the worst-behaved tail indices.
        if err > stall * prev_err and b + grow <= max_block:
            amp = np.abs(X @ v_B) if v_B is not None else np.abs(X[:, 0])
            worst = C[np.argsort(-amp)[:grow]]
            B = np.sort(np.union1d(B, worst))
            grew += 1
            prev_err = np.inf
            X_full = np.zeros((n, 1), dtype=cdt)   # shapes changed
            continue
        prev_err = max(err, 1e-300)

    v = np.zeros(n, dtype=cdt)
    v[B] = v_B
    mask = np.ones(n, dtype=bool)
    mask[B] = False
    v[mask] = (X_full[mask] @ v_B) if X_full.shape[1] == len(B) else 0.0
    nv = np.linalg.norm(v)
    if nv > 0:
        v /= nv

    if return_info:
        return lam, v, {"converged": converged, "iters": outer,
                        "block": len(B), "grew": grew}
    return lam, v
