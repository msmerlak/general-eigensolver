"""Tests for the normal-matrix solver and the norm-reducing shear."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import normal_eig, normality_defect, shear_toward_normal  # noqa: E402


def normal_matrix(n, seed=3):
    """Q (block-diagonal 2x2 rotations) Q^T: normal, nonsymmetric, complex
    spectrum, and a DEGENERATE Hermitian part on every conjugate pair."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    B = np.zeros((n, n))
    for i in range(0, n - 1, 2):
        a, b = rng.standard_normal(2)
        B[i, i] = B[i + 1, i + 1] = a
        B[i, i + 1], B[i + 1, i] = b, -b
    if n % 2:
        B[-1, -1] = rng.standard_normal()
    return Q @ B @ Q.T


def match_err(a, b, scale):
    b = list(b)
    worst = 0.0
    for x in a:
        d = [abs(x - y) for y in b]
        k = int(np.argmin(d))
        worst = max(worst, d[k])
        b.pop(k)
    return worst / scale


def test_normal_eig_exact():
    for n in (40, 61):
        A = normal_matrix(n)
        assert normality_defect(A) < 1e-14
        w, U, info = normal_eig(A, return_info=True)
        scale = np.linalg.norm(A, 2)
        assert info["off"] < 1e-11
        assert match_err(w, np.linalg.eigvals(A), scale) < 1e-12
        assert np.linalg.norm(U.conj().T @ U - np.eye(n), "fro") < 1e-11
        assert np.linalg.norm(A @ U - U * w, "fro") / scale < 1e-11


def test_normal_eig_needs_generic_alpha():
    """alpha=0 diagonalizes only the Hermitian part, which is degenerate on
    every complex pair -- the case the solver exists for."""
    A = normal_matrix(40)
    _, _, bad = normal_eig(A, alpha=0.0, return_info=True)
    _, _, good = normal_eig(A, return_info=True)
    assert bad["off"] > 1e-3
    assert good["off"] < 1e-11


def test_normal_eig_on_symmetric_input():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((40, 40))
    A = (M + M.T) / 2.0
    w, U, info = normal_eig(A, return_info=True)
    assert info["off"] < 1e-11
    assert np.max(np.abs(w.imag)) < 1e-11 * np.linalg.norm(A, 2)


def test_shear_reduces_normality_defect():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 50)) / np.sqrt(50)
    d0 = normality_defect(A)
    B, info = shear_toward_normal(A, max_iter=60)
    assert info["defect"] < d0 / 5
    # a similarity must preserve the spectrum
    assert match_err(np.linalg.eigvals(B), np.linalg.eigvals(A),
                     np.linalg.norm(A, 2)) < 1e-8


def test_shear_leaves_normal_input_alone():
    A = normal_matrix(40)
    B, info = shear_toward_normal(A, max_iter=50)
    assert info["defect"] < 1e-13
    assert np.allclose(B, A, atol=1e-10)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
