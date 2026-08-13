"""Tests for IPT and the SSJ->IPT hybrid. Run with pytest, or directly:
python3 tests/test_ipt.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import ipt_eigh, ipt_eig, ssj_ipt_eigh  # noqa: E402
from ssj.ipt import ipt_rate  # noqa: E402


def near_diagonal(n, ratio, seed=0):
    rng = np.random.default_rng(seed)
    d = np.arange(n, dtype=float)
    W = rng.standard_normal((n, n))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    W *= ratio / np.max(np.abs(W))
    return np.diag(d) + W


def goe(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2.0 * n)


def check(A, w, V, dlam_tol=1e-12, resid_tol=1e-11, ortho_tol=1e-10):
    n = A.shape[0]
    norm2 = np.linalg.norm(A, ord=2)
    assert np.all(np.diff(w) >= 0), "eigenvalues not sorted"
    assert np.max(np.abs(w - np.linalg.eigvalsh(A))) / norm2 < dlam_tol
    assert np.linalg.norm(A @ V - V * w, "fro") / norm2 < resid_tol
    assert np.linalg.norm(V.conj().T @ V - np.eye(n), "fro") < ortho_tol


def test_ipt_near_diagonal():
    A = near_diagonal(120, 0.01)
    w, V, info = ipt_eigh(A, return_info=True)
    assert info["converged"] and info["iters"] <= 10
    check(A, w, V)


def test_ipt_reports_divergence_outside_basin():
    # GOE is far outside IPT's basin: the failure must be reported, not hidden
    w, V, info = ipt_eigh(goe(80, seed=1), return_info=True)
    assert not info["converged"]


def test_ipt_rate_predicts_convergence():
    for ratio, expect in [(0.01, True), (5.0, False)]:
        A = near_diagonal(60, ratio)
        r = ipt_rate(A)
        _, _, info = ipt_eigh(A, return_info=True)
        assert (r < 0.5) == expect
        assert info["converged"] == expect


def test_ipt_rate_infinite_on_exact_ties():
    A = np.diag([1.0, 1.0, 2.0])
    A[0, 1] = A[1, 0] = 0.1
    assert not np.isfinite(ipt_rate(A))


def test_hybrid_global_on_goe():
    A = goe(120, seed=2)
    w, V, info = ssj_ipt_eigh(A, return_info=True)
    assert info["converged"]
    check(A, w, V)


def test_hybrid_takes_pure_ipt_path_when_near_diagonal():
    A = near_diagonal(100, 0.01)
    w, V, info = ssj_ipt_eigh(A, return_info=True)
    assert info["path"] == "ipt" and info["sweeps"] == 0
    check(A, w, V)


def test_hybrid_handles_exact_degeneracies():
    # clustered spectrum: the gate opens but IPT cannot resolve the cluster,
    # so the hybrid must fall back to SSJ rather than return a wrong answer
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((60, 60)))
    vals = np.repeat(rng.uniform(-1, 1, 12), 5)
    A = (Q * vals) @ Q.T
    A = (A + A.T) / 2.0
    w, V, info = ssj_ipt_eigh(A, return_info=True)
    assert info["converged"]
    check(A, w, V)


def test_hybrid_warm_start():
    A = goe(100, seed=4)
    _, V0 = np.linalg.eigh(A)
    rng = np.random.default_rng(5)
    P = rng.standard_normal((100, 100))
    P = (P + P.T) / 2.0
    A2 = A + 1e-5 * P / np.linalg.norm(P, 2)
    w, V, info = ssj_ipt_eigh(A2, X0=V0, return_info=True)
    assert info["converged"] and info["sweeps"] == 0
    check(A2, w, V)


def test_ipt_complex_hermitian():
    rng = np.random.default_rng(6)
    n = 60
    W = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    W = (W + W.conj().T) / 2.0
    np.fill_diagonal(W, 0.0)
    A = np.diag(np.arange(n, dtype=float)) + 0.01 * W / np.max(np.abs(W))
    w, V, info = ipt_eigh(A, return_info=True)
    assert info["converged"]
    check(A, w, V)


def test_ipt_general_nonsymmetric():
    """IPT on a general (nonsymmetric) near-diagonal matrix."""
    rng = np.random.default_rng(10)
    n = 80
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    A = np.diag(np.arange(n, dtype=float)) + 0.01 * W / np.max(np.abs(W))
    w, V, info = ipt_eig(A, return_info=True)
    assert info["converged"] and info["iters"] <= 8
    norm2 = np.linalg.norm(A, ord=2)
    # eigenvalues match LAPACK (spectrum is real in this regime)
    assert np.max(np.abs(np.sort(w.real) - np.sort(np.linalg.eigvals(A).real))) \
        / norm2 < 1e-12
    assert np.max(np.abs(w.imag)) / norm2 < 1e-12
    # residual, with NO orthogonality expected of V
    assert np.linalg.norm(A @ V - V * w, "fro") / norm2 < 1e-11


def test_ipt_general_does_not_orthogonalize():
    """Eigenvectors of a nonsymmetric matrix are not orthogonal; the solver
    must not force them to be (that would be a wrong answer, not a slow one)."""
    rng = np.random.default_rng(11)
    n = 60
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    A = np.diag(np.arange(n, dtype=float)) + 0.2 * W / np.max(np.abs(W))
    w, V, info = ipt_eig(A, return_info=True)
    assert info["converged"]
    off_orth = np.linalg.norm(V.conj().T @ V - np.eye(n), "fro")
    assert off_orth > 1e-6, "V came back orthogonal; symmetry was assumed somewhere"
    assert np.linalg.norm(A @ V - V * w, "fro") / np.linalg.norm(A, 2) < 1e-11


def test_ipt_general_reports_divergence():
    rng = np.random.default_rng(12)
    A = rng.standard_normal((60, 60))  # Ginibre: far outside the basin
    _, _, info = ipt_eig(A, return_info=True)
    assert not info["converged"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
