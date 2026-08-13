"""The dispatching partial solver.

The property that matters is NOT that the screen is accurate -- it is a
one-hop heuristic and is measurably optimistic -- but that a wrong screen
costs time and never correctness.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import eig_partial  # noqa: E402


def band_plus_impurities(n, niso=4, coupling=0.05, seed=0):
    rng = np.random.default_rng(seed)
    nb = n - niso
    d = np.concatenate([rng.uniform(0, 1, nb), np.linspace(-9, 9, niso)])
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    W *= coupling / np.max(np.abs(W))
    return np.diag(d) + W, list(range(nb, n))


def err(A, w, V):
    exact = np.linalg.eigvals(A)
    scale = np.linalg.norm(A, 2)
    e = max(np.min(np.abs(exact - x)) / scale for x in w)
    r = max(np.linalg.norm(A @ V[:, j] - w[j] * V[:, j]) / scale
            for j in range(len(w)))
    return e, r


def test_routes_isolated_targets_to_ipt():
    A, iso = band_plus_impurities(400)
    w, V, info = eig_partial(A, cols=iso, return_info=True)
    assert info["path"] == "ipt" and info["n_ipt"] == len(iso)
    e, r = err(A, w, V)
    assert e < 1e-12 and r < 1e-11


def test_routes_hopeless_targets_to_arpack():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((300, 300))          # Ginibre: no column in basin
    w, V, info = eig_partial(A, sigma=0.0, k=4, return_info=True)
    assert info["path"] == "arpack" and info["n_ipt"] == 0
    e, _ = err(A, w, V)
    assert e < 1e-10


def test_mixed_routing_in_one_call():
    A, iso = band_plus_impurities(400)
    cols = [0, 1] + iso[:2]                      # band columns + impurities
    w, V, info = eig_partial(A, cols=cols, return_info=True)
    assert info["path"] == "mixed"
    assert info["n_ipt"] == 2 and info["n_arpack"] == 2
    e, r = err(A, w, V)
    assert e < 1e-12 and r < 1e-11


def test_optimistic_screen_costs_time_not_correctness():
    """The screen is a one-hop heuristic and CAN pass on a target IPT fails.
    When that happens the result must still be correct."""
    rng = np.random.default_rng(0)
    n = 400
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    W *= 2.0 / np.max(np.abs(W))
    A = np.diag(np.concatenate([rng.uniform(0, 1, n - 1), [9.0]])) + W
    # force IPT even though it will diverge here
    w, V, info = eig_partial(A, cols=[n - 1], gate=1e9, return_info=True)
    e, r = err(A, w, V)
    assert e < 1e-10 and r < 1e-9, "unconverged IPT output must not be returned"


def test_sigma_targeting():
    A, iso = band_plus_impurities(400)
    w, V, info = eig_partial(A, sigma=9.0, k=1, return_info=True)
    assert abs(w[0].real - 9.0) < 0.5
    e, r = err(A, w, V)
    assert e < 1e-12


def _sparse_case(coupling, seed=5, n=2000):
    import scipy.sparse as sp
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from bench_sparse import sparse_diagonally_dominant
    A0, d = sparse_diagonally_dominant(n, seed=seed)
    A = (sp.diags(d) + coupling * (A0 - sp.diags(d))).tocsr()
    return A, list(np.argsort(np.abs(d - np.median(d)))[:4])


def test_accepts_sparse_input():
    """The largest margin in the repository is on large sparse interior
    targets, so the router has to accept sparse input without densifying it."""
    A, cols = _sparse_case(1.0)
    w, V, info = eig_partial(A, cols=cols, return_info=True)
    assert info["path"] == "ipt"
    nrm = float(np.max(np.abs(A.diagonal()))) + 1.0
    assert np.max(np.linalg.norm(A @ V - V * w, axis=0)) / nrm < 1e-12


def test_gate_defaults_by_regime():
    """A wasted attempt costs ~0.4% of the fallback when sparse and several
    times the fallback when dense, so one gate cannot serve both."""
    from ssj.dispatch import _auto_gate
    A_sparse, _ = _sparse_case(1.0)
    A_dense, _ = band_plus_impurities(60)
    assert _auto_gate(A_sparse) == np.inf     # try everything: waste is free
    assert _auto_gate(A_dense) == 0.1         # conservative: waste is not


def test_sparse_mixed_and_hopeless_routing_stay_correct():
    """Both sparse paths, against dense ground truth: a batch where only some
    targets converge must fall back on exactly those that did not, and a batch
    where none do must come back entirely from the fallback -- correct either
    way, which is the property the router exists for."""
    for coupling, want_path, want_ipt in ((80.0, "mixed", 3), (320.0, "arpack", 0)):
        A, cols = _sparse_case(coupling)
        Ad = np.asarray(A.todense())
        ev = np.linalg.eigvalsh(Ad)
        scale = np.linalg.norm(Ad, 2)
        w, V, info = eig_partial(A, cols=cols, return_info=True, max_iter=400)
        assert info["path"] == want_path, (coupling, info["path"])
        assert info["n_ipt"] == want_ipt
        for x in np.real(w):
            assert np.min(np.abs(ev - x)) / scale < 1e-12
        assert np.max(np.linalg.norm(A @ V - V * w, axis=0)) / scale < 1e-11


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
