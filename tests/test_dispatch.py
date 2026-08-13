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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
