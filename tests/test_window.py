"""Interval eigensolving via purification: the certified-count property is
the point, not raw speed (see module docstring for the measured comparison)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj.window import window_count, window_eig  # noqa: E402


def goe(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2.0 * n)


def test_exact_count_and_accuracy_on_dense_random():
    A = goe(300, seed=1)
    ev = np.linalg.eigvalsh(A)
    lo, hi = np.quantile(ev, [0.4, 0.6])
    true = ev[(ev >= lo) & (ev <= hi)]
    w, V, info = window_eig(A, lo, hi, return_info=True)
    assert info["count"] == len(true) == len(w)
    scale = np.linalg.norm(A, 2)
    assert np.max(np.abs(np.sort(w) - np.sort(true))) / scale < 1e-9
    assert np.max(np.linalg.norm(A @ V - V * w, axis=0)) / scale < 1e-8
    assert np.linalg.norm(V.conj().T @ V - np.eye(len(w)), "fro") < 1e-8


def test_empty_window_below_and_above_spectrum():
    A = goe(150, seed=2)
    ev = np.linalg.eigvalsh(A)
    assert window_count(A, ev[0] - 10, ev[0] - 5) == 0
    assert window_count(A, ev[-1] + 5, ev[-1] + 10) == 0
    w, V = window_eig(A, ev[0] - 10, ev[0] - 5)
    assert len(w) == 0 and V.shape == (150, 0)


def test_full_spectrum_window():
    A = goe(120, seed=3)
    ev = np.linalg.eigvalsh(A)
    w, V, info = window_eig(A, ev[0] - 1, ev[-1] + 1, return_info=True)
    assert info["count"] == 120
    scale = np.linalg.norm(A, 2)
    assert np.max(np.abs(np.sort(w) - ev)) / scale < 1e-9


def test_degenerate_eigenvalues_inside_window():
    n = 150
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    vals = np.sort(np.concatenate([[2.0, 2.0, 2.0], np.linspace(-1, 1, n - 3)]))
    A = (Q * vals) @ Q.T
    A = (A + A.T) / 2.0
    w, V, info = window_eig(A, 1.9, 2.1, return_info=True)
    assert info["count"] == 3
    scale = np.linalg.norm(A, 2)
    assert np.max(np.abs(np.sort(w) - 2.0)) / scale < 1e-9
    assert np.max(np.linalg.norm(A @ V - V * w, axis=0)) / scale < 1e-8


def test_window_count_matches_dense_diagonalization_broadly():
    """Sweep several random windows and require exact agreement with a full
    dense eigh -- this is the correctness property the module exists for."""
    rng = np.random.default_rng(5)
    A = goe(200, seed=6)
    ev = np.linalg.eigvalsh(A)
    for _ in range(6):
        lo, hi = np.sort(rng.uniform(ev[0], ev[-1], 2))
        true = int(np.sum((ev >= lo) & (ev <= hi)))
        assert window_count(A, lo, hi) == true




def test_mixed_precision_matches_full_precision():
    A = goe(250, seed=7)
    ev = np.linalg.eigvalsh(A)
    lo, hi = np.quantile(ev, [0.3, 0.7])
    w1, V1, i1 = window_eig(A, lo, hi, precision="full", return_info=True)
    w2, V2, i2 = window_eig(A, lo, hi, precision="mixed", return_info=True)
    assert i1["count"] == i2["count"]
    scale = np.linalg.norm(A, 2)
    assert np.max(np.abs(np.sort(w1) - np.sort(w2))) / scale < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
