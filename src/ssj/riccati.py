"""IPT read as a Riccati equation, and the map that falls out of it.

Pin the target coordinate (v_j = 1) and let x be the tail on C = {i != j}.
The eigenproblem is then EXACT, with no truncation anywhere:

    lambda(x) = a_jj + w^T x                        (row j,  w = A[j, C])
    R(x)      = a_Cj + A_CC x - lambda(x) x = 0     (rows C)

an algebraic Riccati equation. Expanding a step is also exact:

    R(x + delta) = R + J delta - (w^T delta) delta,
    J = A_CC - lambda I - x w^T

so R is a genuine QUADRATIC in delta -- nothing is dropped in writing this.
IPT is the fixed-point iteration that keeps only diag(A_CC) and discards both
the rank-one term x w^T and the quadratic term (w^T delta) delta. Each of
those is a different map, and neither costs an extra matvec to restore.

The one that matters is restoring both, which means solving the local
quadratic model exactly. Put s = w^T delta; then

    (J - s I) delta = -R

and with A_CC replaced by its diagonal, J - sI is diagonal-plus-rank-one, so
each candidate s costs one Sherman-Morrison apply, O(n), and NO matvec. The
scalar equation s = w^T delta(s) is solved by a few damped fixed-point steps.

**s is exactly the eigenvalue increment**, lambda(x + delta) - lambda(x). So
the denominators d_i - (lambda + s) use the UPDATED eigenvalue: this is
self-consistent Brillouin-Wigner perturbation theory, where IPT is the
Rayleigh-Schrodinger form. That is why it helps, and it says exactly WHERE it
helps. IPT diverges when some d_i sits on top of lambda, because it divides
by d_i - lambda with lambda frozen at the current estimate. Brillouin-Wigner
lets lambda move away first -- level repulsion is built into the denominator
-- so the near-degenerate term stays finite.

Measured against this repository's own `ipt_eig_partial` on identical inputs,
with BOTH the eigenvalue and the residual checked against a dense ground
truth. 240 instances, 4 families x 5 couplings x 4 seeds x n in {120,200,300}:

    ipt_eig_partial  70/240        bw_eig_partial  106/240
    union of the two 108/240       BW fails where IPT succeeds: 2/240
    where both solve: median iteration ratio BW/IPT = 1.00

By family (IPT -> BW): symmetric 18 -> 28, degenerate 8 -> 19, graded
25 -> 35, nonsymmetric 19 -> 24. Reproduce with `python bench_riccati.py`.

The degenerate row identifies the mechanism: a near-degenerate partner beside
the target is the single-small-denominator failure, and it more than doubles.

It is NOT a strict superset, and that was worth finding out rather than
assuming -- a first sweep at one size showed zero regressions and a second at
another size found them. In 2 of 240 instances IPT converges and BW does not,
because the scalar equation s = w.delta(s) has several roots and the damped
iterate can settle on one whose denominators have crossed a level. A solver
that must never regress should therefore try IPT on BW's failures; that
recovers both cases (union 108) and costs one extra cheap run. Clamping s to
stay short of the nearest level does NOT work as a fix: it also blocks the
level repulsion the method depends on, and measured, it dropped BW from
49/120 to 32/120 while ADDING regressions.

WHEN TO USE IT, which is narrower than the robustness number alone suggests,
because iteration count is not wall time:

  * DENSE is its home. The matvec is O(n^2 k) there, so the extra O(nk)
    elementwise work amortizes away: measured 1.31x of IPT at n=800 and 0.93x
    (i.e. FASTER) at n=1500, while converging on cases IPT cannot.
  * SPARSE is not. The matvec is only O(nnz), so the inner loop -- about
    24 n-by-k passes against the matvec's ~8 at nnz = 8n -- dominates the
    clock: 2.8x to 6.9x slower than IPT even when both converge.

WHAT IT DOES NOT FIX, measured rather than assumed: strong global coupling.
On the sparse random-graph family of bench_sparse.py it fails at exactly the
same coupling as plain IPT (converges at rho = 0.037, diverges at 0.073).
That failure is many-path accumulation, not a single small denominator, and
Brillouin-Wigner only repairs the latter. Blocks are the fix for the former
(`sparse_block_ipt_eig`). So the two are complementary, and together they say
the sparse recommendation plainly: keep IPT, escalate to blocks, not to this.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bw_eig_partial"]


def _issparse(A):
    return hasattr(A, "tocsr") and hasattr(A, "nnz")


def _rowdot(Arows, Y):
    """For each column m, the inner product of A's row cols[m] with Y[:, m].

    Costs O(nnz of those k rows), never a full matvec, because only the k
    pinned rows are involved. The pinned entries of Y are zero on every call
    below, which is what removes the diagonal term a_jj from w.
    """
    if _issparse(Arows):
        return np.asarray(Arows.multiply(Y.T).sum(axis=1)).ravel()
    return np.einsum("mi,im->m", Arows, Y)


def bw_eig_partial(A, cols, tol=1e-13, max_iter=200, inner=4, hermitian=False,
                   patience=12, return_info=False):
    """k targeted eigenpairs by the self-consistent Brillouin-Wigner map.

    Same interface, same cost structure and same column-separability as
    `ipt_eig_partial` -- one matvec block per iteration, everything else O(nk)
    -- but with the rank-one and quadratic terms of the Riccati residual put
    back, which roughly doubles the set of problems that converge (see the
    module docstring for the measurement).

    Parameters
    ----------
    A : (n, n) dense array or scipy.sparse matrix.
    cols : target column indices; column m converges to the eigenpair whose
        eigenvalue is near A[cols[m], cols[m]].
    inner : damped scalar fixed-point steps for s per outer iteration. Each is
        O(nk) and costs no matvec. The scalar loop is what does the work:
        inner=1 solves 30/120 of the battery below (barely above IPT's 25),
        while inner>=2 reaches 39-49. Default 4.

    Returns (w, V), or (w, V, info) with per-column `converged_cols`,
    `err_cols`, `iters_cols` and `failed`, exactly as `ipt_eig_partial` does.
    """
    sparse = _issparse(A)
    if sparse:
        A = A.tocsr()
        n = A.shape[0]
        d = np.asarray(A.diagonal()).ravel().astype(np.float64)
        scale = float(np.sqrt(A.multiply(A).sum())) / max(np.sqrt(n), 1.0)
        dtype = np.float64 if A.dtype.kind == "f" else A.dtype
    else:
        A = np.asarray(A)
        if A.dtype.kind not in "cf":
            A = A.astype(np.float64)
        n = A.shape[0]
        d = np.diag(A).copy()
        scale = float(np.linalg.norm(A, ord="fro")) / max(np.sqrt(n), 1.0)
        dtype = A.dtype
    if scale == 0.0:
        scale = 1.0

    cols = np.asarray(cols, dtype=int)
    k = len(cols)
    rows = np.arange(k)
    Arows = A[cols]                       # only the k pinned rows are needed
    tol_abs = tol * scale

    V = np.zeros((n, k), dtype=dtype)
    V[cols, rows] = 1.0
    gap = np.empty((n, k), dtype=dtype)          # reused every inner step
    minv = np.empty_like(gap)
    Mr = np.empty_like(gap)
    Mx = np.empty_like(gap)
    delta = np.zeros_like(gap)
    X = np.empty_like(gap)
    conv = np.zeros(k, dtype=bool)
    err_col = np.full(k, np.inf)
    iters_col = np.zeros(k, dtype=int)
    lam = d[cols].astype(dtype, copy=True)
    lim = None
    ref = None
    stalled = np.zeros(k, dtype=bool)
    it = 0

    for it in range(1, max_iter + 1):
        AV = A @ V                                   # the only O(nnz k) work
        lam = AV[cols, rows]
        if hermitian:
            lam = np.real(lam)
        R = AV - V * lam                             # Riccati residual
        R[cols, rows] = 0.0                          # pinned rows: zero by def
        emax = float(np.max(np.abs(R)))
        if lim is None:
            lim = 1e3 * max(emax, 1e-300)

        window = it % patience == 0
        if window or emax <= tol_abs or emax > lim or not np.isfinite(emax):
            e = np.max(np.abs(R), axis=0)
            iters_col[~conv] = it
            err_col[~conv] = e[~conv]
            if window:
                if ref is not None:
                    stalled = (e >= ref * (1.0 - 1e-4)) | ~np.isfinite(e)
                ref = e
            newly = (e <= tol_abs) & ~conv
            conv |= newly
            if conv.all():
                break
            if (stalled | (e > lim) | ~np.isfinite(e))[~conv].all():
                break

        # --- solve the local quadratic model: (J - sI) delta = -R, s = w.delta
        # J - sI is diagonal-plus-rank-one, so each s costs O(nk), no matvec.
        # Written with preallocated buffers: on sparse input the matvec is only
        # O(nnz) and this elementwise work is what dominates the wall clock,
        # so an allocating inner loop costs several times the matvec it saves.
        s = np.zeros(k, dtype=lam.dtype)
        np.copyto(X, V)
        X[cols, rows] = 0.0                          # the tail alone
        for _ in range(inner):
            np.subtract(d[:, None], (lam + s)[None, :], out=gap)
            gap[cols, rows] = 1.0                    # neutralize pinned rows
            np.reciprocal(gap, out=minv)
            minv[cols, rows] = 0.0
            np.multiply(minv, R, out=Mr)
            np.negative(Mr, out=Mr)
            np.multiply(minv, X, out=Mx)
            den = 1.0 - _rowdot(Arows, Mx)
            den = np.where(np.abs(den) < 1e-300, 1.0, den)
            np.multiply(Mx, _rowdot(Arows, Mr) / den, out=delta)
            np.add(delta, Mr, out=delta)
            delta[cols, rows] = 0.0
            s_new = _rowdot(Arows, delta)
            if not np.all(np.isfinite(s_new)):
                break
            if np.all(np.abs(s_new - s) <= 1e-14 * (1.0 + np.abs(s))):
                s = s_new
                break
            s = 0.5 * (s + s_new)                    # damped scalar iteration
        if not np.all(np.isfinite(delta)):
            break
        V += delta
        V[cols, rows] = 1.0

    e = np.max(np.abs(R), axis=0)
    fresh = ~conv
    err_col[fresh] = e[fresh]
    iters_col[fresh] = it
    conv |= (e <= tol_abs)

    nrm = np.linalg.norm(V, axis=0, keepdims=True)
    V = V / np.where(nrm > 0, nrm, 1.0)
    if return_info:
        return lam, V, {"iters": it, "converged": bool(conv.all()),
                        "err": float(np.max(err_col)), "converged_cols": conv,
                        "err_cols": err_col, "iters_cols": iters_col,
                        "failed": np.flatnonzero(~conv)}
    return lam, V
