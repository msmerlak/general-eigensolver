"""Davidson: same preconditioner as IPT, accumulated instead of replacing."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import ipt_eig_partial  # noqa: E402
from ssj.davidson import davidson_eig  # noqa: E402


def near_diagonal(n, coupling, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    W *= coupling / np.max(np.abs(W))
    return np.diag(np.arange(n, dtype=float)) + W


def _check(A, lam, v, tol=1e-11):
    exact = np.linalg.eigvalsh(A)
    scale = np.linalg.norm(A, 2)
    assert np.min(np.abs(exact - lam)) / scale < tol
    assert np.linalg.norm(A @ v - lam * v) / scale < tol


def test_converges_where_ipt_diverges():
    n, tgt = 300, 150
    for coupling in (2.0, 8.0):
        A = near_diagonal(n, coupling)
        assert not ipt_eig_partial(A, [tgt], return_info=True,
                                   hermitian=True)[2]["converged"]
        lam, v, info = davidson_eig(A, tgt, return_info=True)
        assert info["converged"]
        _check(A, lam, v)


def test_faster_than_ipt_where_both_work():
    n, tgt = 300, 150
    A = near_diagonal(n, 0.5)
    plain = ipt_eig_partial(A, [tgt], return_info=True, hermitian=True)[2]
    lam, v, info = davidson_eig(A, tgt, return_info=True)
    assert plain["converged"] and info["converged"]
    assert info["iters"] < plain["iters"]
    _check(A, lam, v)


def test_targets_interior_by_diagonal_entry():
    n = 300
    A = near_diagonal(n, 2.0)
    for tgt in (5, 150, 294):
        lam, v, info = davidson_eig(A, tgt, return_info=True)
        assert info["converged"]
        _check(A, lam, v)
        # the eigenvalue found is the one near that diagonal entry
        assert abs(lam - A[tgt, tgt]) < 5.0


def test_reports_failure_rather_than_returning_junk():
    A = near_diagonal(300, 300.0)
    lam, v, info = davidson_eig(A, 150, max_iter=60, return_info=True)
    assert not info["converged"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
