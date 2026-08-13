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
from ssj.ipt import ipt_rate, ipt_rate_columns  # noqa: E402


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


def band_plus_impurities(n, niso=4, coupling=0.05, seed=0):
    """A dense, strongly coupled band plus a few isolated levels far outside
    it -- an impurity level in a band, a defect state in a gap."""
    rng = np.random.default_rng(seed)
    nb = n - niso
    d = np.concatenate([rng.uniform(0, 1, nb), np.linspace(-9, 9, niso)])
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    W *= coupling / np.max(np.abs(W))
    return np.diag(d) + W, list(range(nb, n))


def test_per_column_rate_escapes_the_global_basin():
    """The headline property of the partial solver: IPT's map is
    column-separable, so the BASIN IS PER-COLUMN too. A matrix hopeless for the
    full method still yields its isolated states."""
    A, iso = band_plus_impurities(400)
    assert ipt_rate(A) > 100, "global rate should be far outside the basin"
    assert not ipt_eig(A, return_info=True)[2]["converged"]

    rates = ipt_rate_columns(A, iso)
    assert np.all(rates < 0.1), "isolated columns should be deep in the basin"

    w, V, info = ipt_eig_partial(A, iso, return_info=True)
    assert info["converged"]
    exact = np.linalg.eigvals(A)
    scale = np.linalg.norm(A, 2)
    for j in range(len(iso)):
        assert np.min(np.abs(exact - w[j])) / scale < 1e-12
        assert np.linalg.norm(A @ V[:, j] - w[j] * V[:, j]) / scale < 1e-11


def test_per_column_rate_flags_band_columns_as_hopeless():
    """The screen must reject what it cannot do, not just accept what it can."""
    A, iso = band_plus_impurities(300)
    band_rates = ipt_rate_columns(A, [0, 1, 2, 3])
    assert np.all(band_rates > 1.0)
    _, _, info = ipt_eig_partial(A, [0, 1, 2, 3], return_info=True)
    assert not info["converged"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
