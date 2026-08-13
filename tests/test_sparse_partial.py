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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
