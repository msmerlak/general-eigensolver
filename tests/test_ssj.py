"""Correctness tests for the SSJ eigensolver. Run with pytest, or directly:
python3 tests/test_ssj.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import ssj_eigh  # noqa: E402
from ssj.core import _angles  # noqa: E402


def assert_decomposition(A, method="auto", tol=1e-13, dlam_tol=1e-12,
                         resid_tol=1e-11, ortho_tol=1e-11):
    n = A.shape[0]
    w, V, info = ssj_eigh(A, tol=tol, method=method, return_info=True)
    assert info["converged"], f"did not converge in {info['sweeps']} sweeps"
    assert np.all(np.diff(w) >= 0), "eigenvalues not sorted"
    wt = np.linalg.eigvalsh(A)
    norm2 = np.linalg.norm(A, ord=2)
    assert np.max(np.abs(w - wt)) < dlam_tol * max(norm2, 1.0)
    assert np.linalg.norm(A @ V - V * w, "fro") / max(norm2, 1e-300) < resid_tol
    assert np.linalg.norm(V.conj().T @ V - np.eye(n), "fro") < ortho_tol
    return info


def test_random_symmetric():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((60, 60))
    assert_decomposition((M + M.T) / 2.0)


def test_all_methods_agree():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((80, 80))
    A = (M + M.T) / 2.0
    for method in ["auto", "qr", "cholqr2", "gemm"]:
        assert_decomposition(A, method=method)


def test_complex_hermitian():
    rng = np.random.default_rng(2)
    M = rng.standard_normal((50, 50)) + 1j * rng.standard_normal((50, 50))
    A = (M + M.conj().T) / 2.0
    for method in ["auto", "gemm"]:
        assert_decomposition(A, method=method)


def test_zero_gap_2x2():
    # every gap is zero: the angle must saturate at pi/4, not vanish
    A = np.array([[0.0, 3.0], [3.0, 0.0]])
    info = assert_decomposition(A)
    assert info["sweeps"] <= 6


def test_exact_degeneracies():
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((60, 60)))
    vals = np.concatenate([np.repeat(rng.uniform(-1, 1, 6), 5),
                           rng.uniform(-1, 1, 30)])
    A = (Q * vals) @ Q.T
    assert_decomposition((A + A.T) / 2.0)


def test_already_diagonal():
    w, V, info = ssj_eigh(np.diag(np.arange(7.0)), return_info=True)
    assert info["sweeps"] == 0
    np.testing.assert_allclose(w, np.arange(7.0))


def test_zero_matrix():
    w, V = ssj_eigh(np.zeros((5, 5)))
    assert np.all(w == 0) and np.allclose(V, np.eye(5))


def test_generator_anti_hermitian_at_ties():
    # exact zero gaps must not break anti-Hermiticity of K (the raw formula
    # gives +pi/4 on both triangles there; orientation is explicit)
    rng = np.random.default_rng(4)
    M = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    B = (M + M.conj().T) / 2.0
    B[np.diag_indices(8)] = 2.0
    K = _angles(B)
    assert np.linalg.norm(K + K.conj().T) == 0.0
    assert np.max(np.abs(K)) <= np.pi / 4 + 1e-15


def test_graded_spectrum_small_eigenvalues():
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((40, 40)))
    A = (Q * 2.0 ** (-np.arange(40, dtype=float))) @ Q.T
    assert_decomposition((A + A.T) / 2.0)


def test_gemm_orthogonality_is_real():
    # regression: with factor-form retraction (X @ orth(I+K)) the gemm variant
    # converges in apparent off(B) while X drifts O(1) from orthogonality;
    # the product form must keep the true orthogonality error at roundoff
    rng = np.random.default_rng(6)
    M = rng.standard_normal((150, 150))
    A = (M + M.T) / np.sqrt(300.0)
    w, V, info = ssj_eigh(A, method="gemm", return_info=True)
    assert info["converged"]
    assert np.linalg.norm(V.T @ V - np.eye(150), "fro") < 1e-11


def test_warm_start():
    rng = np.random.default_rng(7)
    M = rng.standard_normal((120, 120))
    A = (M + M.T) / np.sqrt(240.0)
    _, V0 = np.linalg.eigh(A)
    P = rng.standard_normal((120, 120))
    P = (P + P.T) / 2.0
    A2 = A + 1e-4 * P / np.linalg.norm(P, 2)
    w, V, info = ssj_eigh(A2, X0=V0, return_info=True)
    assert info["converged"] and info["sweeps"] <= 5
    wt = np.linalg.eigvalsh(A2)
    assert np.max(np.abs(w - wt)) < 1e-12
    assert np.linalg.norm(V.T @ V - np.eye(120), "fro") < 1e-11


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
