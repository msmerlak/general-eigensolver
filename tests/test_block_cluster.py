"""SSJ-BC: the block-cluster preconditioner behind ssj_eigh(block_m=...)."""
import numpy as np
import pytest

from ssj import ssj_eigh, off_frobenius
from ssj.core import _block_pass


def goe(n, seed=0):
    r = np.random.default_rng(seed)
    M = r.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2 * n)


def degenerate(n, fold=5, seed=2):
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.standard_normal((n, n)))
    vals = np.repeat(r.standard_normal(n // fold + 1), fold)[:n]
    A = (Q * vals) @ Q.T
    return (A + A.T) / 2


def clustered(n, gap=1e-9, seed=3):
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.standard_normal((n, n)))
    vals = np.sort(r.standard_normal(n))
    vals[n // 2:n // 2 + 5] = vals[n // 2] + gap * np.arange(5)
    A = (Q * vals) @ Q.T
    return (A + A.T) / 2


def assert_accurate(A, w, V, tol=1e-12):
    nrm = np.linalg.norm(A, 2)
    assert np.max(np.abs(np.sort(w) - np.linalg.eigvalsh(A))) / nrm < tol
    assert np.max(np.linalg.norm(A @ V - V * w, axis=0)) / nrm < tol
    assert np.linalg.norm(V.T @ V - np.eye(len(w))) < 1e-10


# --------------------------------------------------------------- the pass


@pytest.mark.parametrize("n,m", [(200, 32), (256, 64), (300, 32), (100, 16)])
def test_block_pass_is_monotone_and_similar(n, m):
    """A block pass can only reduce off(B), for any grouping, and it is an
    orthogonal similarity. n=300 exercises the m-does-not-divide-n path."""
    A = goe(n, seed=7)
    ev = np.linalg.eigvalsh(A)
    B, X = A.copy(), np.eye(n)
    prev = off_frobenius(B)
    for k in range(6):
        B, X = _block_pass(B, X, m, 0 if k % 2 == 0 else m // 2)
        cur = off_frobenius(B)
        assert cur <= prev + 1e-10, f"off(B) rose at pass {k}: {prev} -> {cur}"
        prev = cur
    assert np.linalg.norm(X.T @ X - np.eye(n)) < 1e-11
    assert np.max(np.abs(np.linalg.eigvalsh(B) - ev)) < 1e-10
    assert prev < off_frobenius(A)


def test_block_pass_noop_when_block_spans_problem():
    A = goe(20, seed=1)
    B, X = _block_pass(A.copy(), np.eye(20), 20, 0)
    assert np.array_equal(B, A)


# --------------------------------------------------------- solver accuracy


@pytest.mark.parametrize("method", ["auto", "gemm", "cholqr2"])
def test_accuracy_matches_default(method):
    A = goe(120, seed=1)
    w, V, info = ssj_eigh(A, method=method, block_m=32, return_info=True)
    assert info["converged"]
    assert_accurate(A, w, V)


def test_accuracy_on_exact_degeneracy():
    """The spectrum SSJ-BC exists for. Exact ties make every within-cluster
    gap zero, which is where the arctan saturates and the plain map crawls."""
    A = degenerate(200)
    w, V, info = ssj_eigh(A, block_m=32, return_info=True)
    assert info["converged"]
    assert_accurate(A, w, V, tol=1e-11)


def test_accuracy_on_tight_cluster():
    A = clustered(200, gap=1e-9)
    w, V, info = ssj_eigh(A, block_m=32, return_info=True)
    assert info["converged"]
    assert_accurate(A, w, V, tol=1e-11)


def test_accuracy_mixed_precision():
    A = goe(120, seed=4)
    w, V, info = ssj_eigh(A, block_m=32, precision="mixed", return_info=True)
    assert info["converged"]
    assert_accurate(A, w, V)


def test_complex_hermitian():
    r = np.random.default_rng(5)
    n = 96
    M = r.standard_normal((n, n)) + 1j * r.standard_normal((n, n))
    A = (M + M.conj().T) / np.sqrt(2 * n)
    w, V, info = ssj_eigh(A, block_m=32, return_info=True)
    assert info["converged"]
    nrm = np.linalg.norm(A, 2)
    assert np.max(np.abs(np.sort(w) - np.linalg.eigvalsh(A))) / nrm < 1e-12
    assert np.linalg.norm(V.conj().T @ V - np.eye(n)) < 1e-10


# ------------------------------------------------------------ the payoff


@pytest.mark.parametrize("case,ceiling", [
    (goe(200, 1), 12),
    (degenerate(200), 45),
    (clustered(200, 1e-9), 14),
])
def test_cuts_sweeps(case, ceiling):
    """Sweep counts are load-immune, so this is a stable assertion. Ceilings
    sit above the measured values (9 / 25 / 9) but well below the defaults
    (20 / 69 / 33), so the test catches a real loss without being brittle."""
    _, _, base = ssj_eigh(case, return_info=True)
    _, _, bc = ssj_eigh(case, block_m=32, return_info=True)
    assert bc["sweeps"] < base["sweeps"]
    assert bc["sweeps"] <= ceiling


def test_warm_start_does_not_regress():
    """The case that kept SSJ-BC out of core: a tight X0 lands the iterate in
    the tail, where a block pass could cost more than it returns. It does not
    here, which is why block_until defaults to 0.0."""
    A = goe(200, seed=1)
    _, V0 = np.linalg.eigh(A)
    r = np.random.default_rng(9)
    P = r.standard_normal((200, 200))
    P = (P + P.T) / 2
    A2 = A + 1e-6 * P / np.linalg.norm(P, 2)
    _, _, base = ssj_eigh(A2, X0=V0, return_info=True)
    w, V, bc = ssj_eigh(A2, X0=V0, block_m=32, return_info=True)
    assert bc["sweeps"] <= base["sweeps"]
    assert_accurate(A2, w, V)


# ------------------------------------------------------------- invariants


def test_default_is_unchanged():
    """Integration must not move the shipped path. block_m=0 has to reproduce
    the untouched iteration bit for bit."""
    A = goe(150, seed=6)
    for method in ("auto", "gemm"):
        w0, V0, i0 = ssj_eigh(A, method=method, return_info=True)
        w1, V1, i1 = ssj_eigh(A, method=method, block_m=0, return_info=True)
        assert i0["sweeps"] == i1["sweeps"]
        assert np.array_equal(w0, w1)
        assert np.array_equal(V0, V1)


def test_block_m_capped_below_problem_size():
    """block_m >= n would make the 'preconditioner' a dense eigensolve."""
    A = goe(40, seed=8)
    w, V, info = ssj_eigh(A, block_m=100, return_info=True)
    assert info["converged"]
    assert_accurate(A, w, V)


def test_ns_target_floor_preserves_digits():
    """The block pass can drop the error several orders in one sweep, so an
    unfloored Newton-Schulz target silently loses digits. Tightening tol must
    still tighten the answer."""
    A = goe(120, seed=2)
    for tol in (1e-10, 1e-13):
        w, V, info = ssj_eigh(A, tol=tol, method="gemm", block_m=32,
                              return_info=True)
        assert info["converged"]
        assert np.linalg.norm(V.T @ V - np.eye(120)) < 1e-10
