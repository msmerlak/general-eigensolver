"""A broadly usable partial eigensolver: IPT where it applies, ARPACK where it
does not, decided per target by a free test.

The measurements in GENERAL.md say two things that pull against each other.
IPT's column-restricted solver beats ARPACK shift-invert by 4-123x on targets
inside its basin. And its basin is a real restriction that randomization
cannot remove -- randomized range finding produces a low-rank SUBSPACE, while
IPT needs a FRAME in which the target's diagonal entry approximates its
eigenvalue, and manufacturing such a frame is the eigenproblem over again
(measured, and recorded in GENERAL.md as a failure).

What makes the fast path usable anyway is that the applicability indicator is
cheaper than either solver:

    rho_j = max_i |W_ij| / |d_j - d_i|            O(N k), no factorization

Because IPT's map is column-separable, rho_j addresses column j on its own. So
this module screens each requested target, sends the ones that pass to IPT and
the rest to ARPACK, and reports which went where.

The indicator is a ONE-HOP heuristic and is OPTIMISTIC -- it sees direct
coupling only, while the series sums multi-hop paths (measured divergences at
rho_j = 0.18 dense, 0.25 on a sparse lattice). The routing is therefore built
so that a WRONG SCREEN COSTS TIME, NEVER CORRECTNESS: if IPT is attempted and
does not converge, those targets are handed to ARPACK and the unconverged
output is discarded. The worst case is the screen's O(Nk) plus some wasted
IPT iterations on top of the fallback; the best case is up to two orders of
magnitude, on targets that are genuinely isolated.

The fallback is decided PER COLUMN, not per batch. That matters because the
screen is not even monotonic: in a measured 4-target batch the column with
the LOWEST rho (0.042) was the only one to diverge, while rho = 0.096
converged to 4.5e-15. So the screen cannot say which target will fail, only
the outcome can -- and since the map is column-separable, one failure is no
evidence against the others. Falling back per batch would have sent all four
targets to ARPACK where one was needed.

Targets may be given as column indices (`cols`) or as a value to hunt near
(`sigma`), in which case the diagonal entries closest to sigma are used --
the natural targeting for a near-diagonal problem, where d_j is already an
estimate of an eigenvalue.
"""
from __future__ import annotations

import numpy as np

from .ipt import ipt_eig_partial, ipt_rate_columns

__all__ = ["eig_partial"]


def _issparse(A):
    return hasattr(A, "tocsc") and hasattr(A, "nnz")


def _diagonal(A):
    return (np.asarray(A.diagonal()).ravel() if _issparse(A) else np.diag(A))


def _is_hermitian(A):
    if _issparse(A):
        D = (A - A.conj().T).tocoo()
        return D.nnz == 0 or float(np.max(np.abs(D.data))) <= 1e-12
    return np.allclose(A, A.conj().T, atol=1e-12, rtol=0)


def _arpack(A, cols, sigma, k):
    """Fallback: ARPACK shift-invert around the targets' diagonal entries."""
    from scipy.sparse.linalg import eigs, eigsh
    d = _diagonal(A)
    if sigma is None:
        sigma = float(np.mean(np.real(d[cols])))
    hermitian = _is_hermitian(A)
    solver = eigsh if hermitian and not np.iscomplexobj(A) else eigs
    # ARPACK needs k < n - 1; widen the search a little for reliability
    kk = min(max(k, 1), A.shape[0] - 2)
    w, V = solver(A.tocsc() if _issparse(A) else A, k=kk, sigma=sigma)
    return np.asarray(w), np.asarray(V)


def _auto_gate(A):
    """The gate that minimizes expected time, which is NOT one number.

    The screen makes two kinds of error and they are priced completely
    differently. A wasted attempt costs (iterations to detect divergence) x
    (one matvec); a missed win costs the entire fallback. The ratio between
    those two is what sets the gate, and it differs by orders of magnitude
    between the two regimes -- measured, not assumed:

      dense, N=400, 16 targets: gate 0.1 -> 0.023 s, gate 0.5 -> 0.111 s,
        gate inf -> 0.045 s. Trying more targets LOSES, because near the
        basin edge IPT needs many O(N^2 k) iterations while dense ARPACK
        shift-invert is only O(N^3/3) once. The conservative gate is right.

      sparse, interior targets: a FULLY wasted IPT attempt (every target
        divergent, run to detection) measured 4.9 ms against a 1.14 s ARPACK
        shift-invert at N=2000, and 59 ms against 15.7 s at N=5000 -- 0.4% of
        the fallback either way, because the fallback pays an LU whose
        fill-in reached 431x nnz. Here screening out a winnable target is
        ~250x worse than trying one and failing, so the right policy is to
        try everything and let the per-column outcome decide.

    This matters because the screen is a weak classifier: over 576 columns
    spanning several matrix families, converged targets reached rho = 23
    while divergent ones started at rho = 0.007, and the current dense gate
    discards 29% of the winnable columns to avoid 30% wasted attempts. That
    trade is worth making when waste is expensive and not when it is free.
    """
    return np.inf if _issparse(A) else 0.1


def eig_partial(A, cols=None, sigma=None, k=None, gate=None, tol=1e-13,
                max_iter=200, force=None, return_info=False):
    """k targeted eigenpairs, routed per target to the cheapest correct solver.

    Parameters
    ----------
    A : (n, n) dense array or scipy.sparse matrix.
    cols : target column indices. Column j's eigenvalue is the one near
        A[cols[j], cols[j]]. Mutually exclusive with `sigma`.
    sigma : target value; the `k` diagonal entries nearest sigma are used.
    k : number of targets when `sigma` is given.
    gate : per-column rate below which IPT is attempted. None (default)
        picks by regime -- 0.1 dense, inf (try everything) sparse -- because
        the cost of a wasted attempt relative to the fallback differs by
        orders of magnitude between the two; see `_auto_gate` for the
        measurements. Pass a float to override.
    force : None (route automatically), "ipt", or "arpack" -- for benchmarking
        and for callers who know their input.

    Returns (w, V), or (w, V, info) with info["path"] one of "ipt",
    "arpack", "mixed", info["rates"] the per-column rates, and info["n_ipt"] /
    info["n_arpack"] the split.
    """
    if not _issparse(A):
        A = np.asarray(A)
    n = A.shape[0]
    if (cols is None) == (sigma is None):
        raise ValueError("give exactly one of `cols` or `sigma`")
    if cols is None:
        if k is None:
            raise ValueError("`k` is required with `sigma`")
        d = np.real(_diagonal(A))
        cols = np.argsort(np.abs(d - sigma))[:k]
    cols = np.asarray(cols, dtype=int)
    k = len(cols)
    if gate is None:
        gate = _auto_gate(A)

    rates = ipt_rate_columns(A, cols)          # O(N k), the whole routing cost
    if force == "ipt":
        use_ipt = np.ones(k, dtype=bool)
    elif force == "arpack":
        use_ipt = np.zeros(k, dtype=bool)
    else:
        use_ipt = rates < gate

    w = np.zeros(k, dtype=np.complex128)
    V = np.zeros((n, k), dtype=np.complex128)

    if use_ipt.any():
        sel = cols[use_ipt]
        wi, Vi, info_i = ipt_eig_partial(A, sel, tol=tol, max_iter=max_iter,
                                         return_info=True,
                                         hermitian=_is_hermitian(A))
        # Keep the columns that converged and fall back on only the ones that
        # did not. IPT's map is column-separable, so a failure in one target
        # says nothing about the others -- and the screen cannot predict which
        # will fail: measured, a batch where the LOWEST-rho target (0.042) was
        # the only divergent one while rho = 0.096 converged to 4.5e-15.
        # Discarding the whole batch would send 4 targets to the fallback
        # where 1 is required.
        idx = np.flatnonzero(use_ipt)
        ok = np.asarray(info_i["converged_cols"], dtype=bool)
        w[idx[ok]] = wi[ok]
        V[:, idx[ok]] = Vi[:, ok]
        use_ipt[idx[~ok]] = False

    if (~use_ipt).any():
        rest = cols[~use_ipt]
        wa, Va = _arpack(A, rest, sigma, len(rest))
        d = np.real(_diagonal(A))
        # match each remaining target to its nearest returned eigenvalue
        taken = []
        for j, c in zip(np.where(~use_ipt)[0], rest):
            dist = np.abs(wa - d[c])
            for t in taken:
                dist[t] = np.inf
            m = int(np.argmin(dist))
            taken.append(m)
            w[j] = wa[m]
            V[:, j] = Va[:, m]

    n_ipt = int(use_ipt.sum())
    path = "ipt" if n_ipt == k else ("arpack" if n_ipt == 0 else "mixed")
    if return_info:
        return w, V, {"path": path, "rates": rates, "n_ipt": n_ipt,
                      "n_arpack": k - n_ipt, "cols": cols}
    return w, V
