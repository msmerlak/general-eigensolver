"""The Brillouin-Wigner map: IPT's Riccati residual with the rank-one and
quadratic terms restored.

The claim being guarded is narrow and specific: same interface and iteration
count as `ipt_eig_partial`, a clear NET gain over it, and a large gain on the
near-degenerate case that is IPT's characteristic failure. It is deliberately
not "no regressions" -- 2 of 240 instances go the other way, so the union of
the two is what loses nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import ipt_eig_partial  # noqa: E402
from ssj.riccati import bw_eig_partial  # noqa: E402


def near_diagonal(n, coupling, seed=0, sym=True):
    rng = np.random.default_rng(seed)
    d = np.sort(rng.uniform(0, 100, n))
    M = rng.standard_normal((n, n))
    if sym:
        M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    M *= coupling * (100.0 / n) / np.max(np.abs(M))
    return np.diag(d) + M


def degenerate(n, gap, coupling=0.2, seed=1):
    """A near-degenerate partner beside the target: the small denominator that
    a Rayleigh-Schrodinger denominator divides by and Brillouin-Wigner does
    not, because there lambda moves off the partner first."""
    A = near_diagonal(n, coupling, seed=seed)
    d = np.diag(A).copy()
    j = n // 2
    d[j + 1] = d[j] + gap
    np.fill_diagonal(A, d)
    return A, j


def test_matches_dense_truth_on_an_easy_problem():
    A = near_diagonal(200, 0.5, seed=3)
    j = 100
    ev = np.linalg.eigvalsh(A)
    scale = np.linalg.norm(A, 2)
    w, V, info = bw_eig_partial(A, [j], return_info=True, hermitian=True)
    assert info["converged"]
    assert np.min(np.abs(ev - w[0])) / scale < 1e-12
    assert np.linalg.norm(A @ V[:, 0] - w[0] * V[:, 0]) / scale < 1e-11


def test_solves_near_degeneracy_where_ipt_fails():
    """The mechanism claim: IPT's denominator sits on a near-degenerate level
    with lambda frozen, BW lets lambda move off it first."""
    A, j = degenerate(200, 1e-6)   # IPT diverges here, BW converges in 32
    ev = np.linalg.eigvalsh(A)
    scale = np.linalg.norm(A, 2)

    _, _, ipt_info = ipt_eig_partial(A, [j], return_info=True, hermitian=True,
                                     max_iter=400)
    w, V, info = bw_eig_partial(A, [j], return_info=True, hermitian=True,
                                max_iter=400)
    assert not ipt_info["converged"]          # IPT genuinely fails here
    assert info["converged"]
    assert np.min(np.abs(ev - w[0])) / scale < 1e-11
    assert np.linalg.norm(A @ V[:, 0] - w[0] * V[:, 0]) / scale < 1e-10


def test_gains_over_ipt_and_the_union_loses_nothing():
    """BW solves substantially more than IPT (70 -> 106 of 240 in
    bench_riccati.py) but is NOT a strict superset: 2 of those 240 go the
    other way. So the property to guard is the one that actually holds --
    a clear net gain, and a union that loses nothing to either alone. This is
    a fast subset of the same sweep, at the size where the regression appears.
    """
    gained = lost = 0
    for coupling in (0.5, 2.0, 8.0):
        for seed in range(3):
            for sym in (True, False):
                A = near_diagonal(120, coupling, seed=seed, sym=sym)
                j = 60
                ev = np.linalg.eigvalsh(A) if sym else np.linalg.eigvals(A)
                scale = np.linalg.norm(A, 2)

                def ok(w, V, info):
                    return bool(info["converged"]
                                and np.min(np.abs(ev - w[0])) / scale < 1e-10)

                a = ok(*ipt_eig_partial(A, [j], return_info=True,
                                        hermitian=sym, max_iter=400))
                b = ok(*bw_eig_partial(A, [j], return_info=True,
                                       hermitian=sym, max_iter=400))
                gained += int(b and not a)
                lost += int(a and not b)
    assert gained > lost                       # a clear net gain ...
    assert gained >= 3                         # ... and not a marginal one


def test_batched_columns_are_independent_and_distinct():
    """Same column-separability as ipt_eig_partial: k targets in one call,
    each converging to its own eigenpair, reported per column."""
    A = near_diagonal(200, 1.0, seed=5)
    cols = [50, 100, 150]
    ev = np.linalg.eigvalsh(A)
    scale = np.linalg.norm(A, 2)
    w, V, info = bw_eig_partial(A, cols, return_info=True, hermitian=True,
                                max_iter=400)
    conv = np.asarray(info["converged_cols"])
    assert conv.any()
    assert len(set(np.round(np.real(w[conv]), 9))) == int(conv.sum())
    for m in np.flatnonzero(conv):
        assert np.min(np.abs(ev - w[m])) / scale < 1e-10
        assert np.linalg.norm(A @ V[:, m] - w[m] * V[:, m]) / scale < 1e-9


def test_accepts_sparse_input():
    """Sparse works, though the module docstring records that it is the wrong
    tool there: the matvec is too cheap to amortize the O(nk) inner loop."""
    import scipy.sparse as sp
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from bench_sparse import sparse_diagonally_dominant
    A, d = sparse_diagonally_dominant(2000, seed=0)
    cols = list(np.argsort(np.abs(d - np.median(d)))[:3])
    w, V, info = bw_eig_partial(A, cols, return_info=True, hermitian=True)
    assert info["converged"]
    scale = float(np.max(np.abs(d))) + 1.0
    assert np.max(np.linalg.norm(A @ V - V * w, axis=0)) / scale < 1e-11


def test_nonsymmetric_vectors_are_not_orthogonal():
    """No symmetry is used anywhere in the map, and a symmetry assumption
    leaking in would show up as orthogonal output."""
    A = near_diagonal(150, 1.0, seed=7, sym=False)
    cols = [40, 75, 110]
    ev = np.linalg.eigvals(A)
    scale = np.linalg.norm(A, 2)
    w, V, info = bw_eig_partial(A, cols, return_info=True, max_iter=400)
    conv = np.asarray(info["converged_cols"])
    assert conv.sum() >= 2
    idx = np.flatnonzero(conv)
    for m in idx:
        assert np.min(np.abs(ev - w[m])) / scale < 1e-10
    G = V[:, idx].conj().T @ V[:, idx]
    off = G - np.diag(np.diag(G))
    assert np.max(np.abs(off)) > 1e-8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
