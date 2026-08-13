"""Few-eigenpair solving by column-restricted IPT.

The property under test is that the IPT map is column-separable, so a k-column
run is EXACT, not an approximation: no deflation, no locking, no loss of
accuracy relative to the full solve.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import ipt_eig, ipt_eig_partial  # noqa: E402


def near_diagonal(n, ratio, seed=0, sym=False):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    if sym:
        W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    return np.diag(np.arange(n, dtype=float)) + ratio * W / np.max(np.abs(W))


def test_partial_matches_full_solve_exactly():
    """Column separability: restricting to k columns must reproduce the same
    eigenpairs the full run produces, to roundoff."""
    A = near_diagonal(200, 0.05)
    w_full, _ = ipt_eig(A, sort=False)
    cols = [3, 77, 100, 101, 199]
    w_part, V = ipt_eig_partial(A, cols)
    for j, c in enumerate(cols):
        assert abs(w_part[j] - w_full[c]) < 1e-11 * np.linalg.norm(A, 2)


def test_partial_targets_interior_eigenvalues():
    """Interior targets cost no more than extremal ones -- the property Krylov
    methods lack without shift-invert."""
    n = 300
    A = near_diagonal(n, 0.05)
    scale = np.linalg.norm(A, 2)
    exact = np.linalg.eigvals(A)
    for cols in ([0, 1], [n // 2, n // 2 + 1], [n - 2, n - 1]):
        w, V, info = ipt_eig_partial(A, cols, return_info=True)
        assert info["converged"] and info["iters"] <= 12
        for j in range(len(cols)):
            assert np.min(np.abs(exact - w[j])) / scale < 1e-11
            assert np.linalg.norm(A @ V[:, j] - w[j] * V[:, j]) / scale < 1e-11


def test_partial_single_eigenpair():
    A = near_diagonal(150, 0.05)
    w, V = ipt_eig_partial(A, [75])
    assert V.shape == (150, 1)
    assert np.linalg.norm(A @ V[:, 0] - w[0] * V[:, 0]) / np.linalg.norm(A, 2) < 1e-11


def test_partial_symmetric():
    A = near_diagonal(200, 0.05, sym=True)
    cols = [50, 100, 150]
    w, V, info = ipt_eig_partial(A, cols, return_info=True, hermitian=True)
    assert info["converged"]
    assert np.max(np.abs(w.imag)) < 1e-12 if np.iscomplexobj(w) else True
    exact = np.linalg.eigvalsh(A)
    for j in range(len(cols)):
        assert np.min(np.abs(exact - w[j])) / np.linalg.norm(A, 2) < 1e-11


def test_partial_reports_divergence():
    """Outside the basin the partial solver must report failure like the full
    one, not return plausible-looking junk."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((100, 100))       # Ginibre, far outside the basin
    _, _, info = ipt_eig_partial(A, [10, 20], return_info=True)
    assert not info["converged"]


def test_partial_cost_scales_with_k_not_n():
    """k columns must cost ~k/N of the full run, not the same."""
    A = near_diagonal(400, 0.05)
    import time
    t0 = time.perf_counter()
    ipt_eig_partial(A, list(range(4)))
    t_small = time.perf_counter() - t0
    t0 = time.perf_counter()
    ipt_eig(A)
    t_full = time.perf_counter() - t0
    assert t_small < t_full / 5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
