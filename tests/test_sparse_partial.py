"""IPT's partial solver on large sparse matrices -- the structural win.

Interior targets on sparse input force any Krylov method into shift-invert,
i.e. a factorization; on sparsity with no good elimination ordering that
factorization is what dominates. IPT needs only matvecs. These tests verify
correctness (against dense ground truth) and the no-factorization property.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench_sparse import sparse_diagonally_dominant  # noqa: E402

from ssj import ipt_eig_partial  # noqa: E402


def test_correct_against_dense_ground_truth():
    """Not merely low-residual: the eigenvalues must be the RIGHT ones, and
    distinct (a solver collapsing all columns onto one eigenvalue would also
    show a small residual)."""
    n = 1000
    A, d = sparse_diagonally_dominant(n, seed=0)
    Ad = np.asarray(A.todense())
    ev = np.linalg.eigvalsh(Ad)
    nrm = np.linalg.norm(Ad, 2)
    cols = list(np.argsort(np.abs(d - np.median(d)))[:4])

    w, V, info = ipt_eig_partial(A, cols, return_info=True, hermitian=True)
    assert info["converged"] and info["iters"] <= 12
    # each returned eigenvalue matches a true one
    for x in w:
        assert np.min(np.abs(ev - x)) / nrm < 1e-12
    # and they are four DISTINCT eigenvalues, not one found four times
    assert len(set(np.round(w, 9))) == 4
    for j in range(len(cols)):
        assert np.linalg.norm(Ad @ V[:, j] - w[j] * V[:, j]) / nrm < 1e-12


def test_accepts_sparse_and_never_densifies():
    """The whole advantage is that no factorization and no dense N-by-N
    intermediate is ever formed. A 20000-square dense array would be 3.2 GB;
    if this test runs at all in reasonable memory, the sparse path is real."""
    n = 20000
    A, d = sparse_diagonally_dominant(n, seed=1)
    assert sp.issparse(A)
    cols = list(np.argsort(np.abs(d - np.median(d)))[:3])
    w, V, info = ipt_eig_partial(A, cols, return_info=True, hermitian=True)
    assert info["converged"]
    assert V.shape == (n, 3)
    nrm = float(np.max(np.abs(d))) + 1.0
    for j in range(3):
        assert np.linalg.norm(A @ V[:, j] - w[j] * V[:, j]) / nrm < 1e-10


def test_iteration_count_does_not_grow_with_N():
    """IPT's cost is O(nnz) per iteration and the iteration count is set by
    the rate, not the size -- so the advantage over a factorization-based
    method widens with N rather than being a fixed constant."""
    counts = []
    for n in (2000, 8000):
        A, d = sparse_diagonally_dominant(n, seed=2)
        cols = list(np.argsort(np.abs(d - np.median(d)))[:4])
        _, _, info = ipt_eig_partial(A, cols, return_info=True, hermitian=True)
        assert info["converged"]
        counts.append(info["iters"])
    assert counts[1] <= counts[0] + 2

def test_nonsymmetric_sparse_correct_and_not_orthogonal():
    """Nonsymmetric sparse interior targets: the case with no LOBPCG
    equivalent, so shift-invert is the only alternative. Verified against
    dense ground truth, and the eigenvectors must come back NON-orthogonal
    (a symmetry assumption leaking in would show up as orthogonal output)."""
    from bench_sparse import sparse_nonsymmetric
    n = 800
    A, d = sparse_nonsymmetric(n, seed=0)
    Ad = np.asarray(A.todense())
    ev = np.linalg.eigvals(Ad)
    nrm = np.linalg.norm(Ad, 2)
    cols = list(np.argsort(np.abs(d - np.median(d)))[:4])
    w, V, info = ipt_eig_partial(A, cols, return_info=True, hermitian=False)
    assert info["converged"]
    for x in w:
        assert np.min(np.abs(ev - x)) / nrm < 1e-12
    assert len(set(np.round(w, 9))) == 4
    for j in range(4):
        assert np.linalg.norm(Ad @ V[:, j] - w[j] * V[:, j]) / nrm < 1e-11


def test_scales_to_large_N():
    """O(nnz) cost with an iteration count set by the rate, not the size."""
    from bench_sparse import sparse_diagonally_dominant
    n = 100000
    A, d = sparse_diagonally_dominant(n, seed=3)
    cols = list(np.argsort(np.abs(d - np.median(d)))[:4])
    w, V, info = ipt_eig_partial(A, cols, return_info=True, hermitian=True)
    assert info["converged"] and info["iters"] <= 8
    nrm = float(np.max(np.abs(d))) + 1.0
    for j in range(4):
        assert np.linalg.norm(A @ V[:, j] - w[j] * V[:, j]) / nrm < 1e-10


def test_many_eigenpairs_stay_cheap_per_pair():
    """Not a handful-of-eigenpairs method: because the columns are
    independent, asking for 128 targets instead of 4 buys no extra coupling
    and no extra iterations -- cost is linear in k at a flat per-pair rate."""
    n = 5000
    A, d = sparse_diagonally_dominant(n, seed=4)
    order = np.argsort(np.abs(d - np.median(d)))
    nrm = float(np.max(np.abs(d))) + 1.0
    its = []
    for k in (4, 128):
        cols = list(order[:k])
        w, V, info = ipt_eig_partial(A, cols, return_info=True, hermitian=True)
        assert info["converged"]
        assert V.shape == (n, k)
        its.append(info["iters"])
        resid = np.max(np.linalg.norm(A @ V - V * w, axis=0)) / nrm
        assert resid < 1e-10
    assert its[1] <= its[0] + 2         # 32x more targets, same iteration count


def test_fails_loudly_outside_the_basin():
    """The dispatcher's safety property: outside the basin the solver must
    REPORT failure, never return a plausible wrong answer. Uses a generous
    max_iter, because divergence here can be slow (763 iterations in the
    GENERAL.md sweep) and a tight budget cannot tell it from slow success."""
    n = 2000
    A0, d = sparse_diagonally_dominant(n, seed=5)
    W = A0 - sp.diags(d)
    cols = list(np.argsort(np.abs(d - np.median(d)))[:3])
    for coupling, expect in ((5.0, True), (320.0, False)):
        A = sp.diags(d) + coupling * W
        _, _, info = ipt_eig_partial(A, cols, return_info=True, hermitian=True,
                                     max_iter=1000)
        assert bool(info["converged"]) is expect, (coupling, info)


def test_rate_screen_accepts_sparse_without_densifying():
    """The screen is recommended precisely for the large-sparse case, so it
    must not build a dense N-by-N intermediate to compute an O(Nk) quantity.
    Values must match the dense computation exactly."""
    from ssj.ipt import ipt_rate_columns
    n = 1500
    A, d = sparse_diagonally_dominant(n, seed=6)
    cols = list(np.argsort(np.abs(d - np.median(d)))[:5])
    r_sparse = ipt_rate_columns(A, cols)
    r_dense = ipt_rate_columns(np.asarray(A.todense()), cols)
    assert np.allclose(r_sparse, r_dense, rtol=1e-12, atol=0)
    # and it is usable at a size where densifying would be 32 GB
    big, dbig = sparse_diagonally_dominant(100000, seed=7)
    rb = ipt_rate_columns(big, [int(np.argmin(np.abs(dbig - np.median(dbig))))])
    assert np.isfinite(rb[0]) and rb[0] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
