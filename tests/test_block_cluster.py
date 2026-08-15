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


def test_batched_and_looped_apply_agree():
    """_block_pass takes the looped branch on numpy and the batched branch on
    cupy. Only the first runs in CI, so pin the second against it here --
    otherwise the GPU path can drift untested."""
    n, m = 128, 32
    r = np.random.default_rng(11)
    M = r.standard_normal((n, n))
    B0 = (M + M.T) / np.sqrt(2 * n)
    X0 = np.linalg.qr(r.standard_normal((n, n)))[0]

    nb, keep = n // m, (n // m) * m
    p = np.argsort(np.diag(B0))
    B, X = B0[p][:, p], X0[:, p]
    idx = np.arange(nb)
    blocks = B[:keep, :keep].reshape(nb, m, nb, m)[idx, :, idx, :]
    Q = np.linalg.eigh((blocks + blocks.transpose(0, 2, 1)) / 2.0)[1]

    Bl, Xl = B.copy(), X.copy()
    for b in range(nb):
        s = slice(b * m, (b + 1) * m)
        Xl[:, s] = Xl[:, s] @ Q[b]
        Bl[:, s] = Bl[:, s] @ Q[b]
        Bl[s, :] = Q[b].T @ Bl[s, :]

    Bb, Xb = B.copy(), X.copy()

    def rmul(Mx):
        return ((Mx.reshape(Mx.shape[0], nb, m).transpose(1, 0, 2) @ Q)
                .transpose(1, 0, 2).reshape(Mx.shape[0], keep))

    Xb[:, :keep] = rmul(Xb[:, :keep])
    Bb[:, :keep] = rmul(Bb[:, :keep])
    Bb[:keep, :] = (Q.transpose(0, 2, 1) @ Bb[:keep, :].reshape(nb, m, n)
                    ).reshape(keep, n)

    assert np.allclose(Xl, Xb, atol=1e-13)
    assert np.allclose(Bl, Bb, atol=1e-13)


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


def test_gemm_on_tight_cluster_keeps_digits():
    """The one combination where the Newton-Schulz target is load-bearing.
    Without any tightening this measures 4.5e-13 eigenvalue error and 4.8e-12
    orthogonality; the predictive rule holds it two orders better, at the cost
    of not matching the always-on floor's 2.9e-15. Pin the trade-off so a
    future change to _ns_target has to face it."""
    A = clustered(200, gap=1e-9)
    w, V, info = ssj_eigh(A, method="gemm", block_m=32, return_info=True)
    assert info["converged"]
    nrm = np.linalg.norm(A, 2)
    assert np.max(np.abs(np.sort(w) - np.linalg.eigvalsh(A))) / nrm < 1e-13
    assert np.linalg.norm(V.T @ V - np.eye(200)) < 1e-11


def test_ns_tightening_is_off_early():
    """The target must stay loose while the error is still large -- that is
    the whole saving. Directly exercises _ns_target rather than inferring it."""
    from ssj.core import _ns_target
    tol = 1e-13
    # early: rel_off O(1), previous sweep contracted by 10x
    assert _ns_target(0.05, 32, tol, 1.0, 10.0) == 0.05
    # late: next sweep predicted at/below tol -> tighten
    assert _ns_target(0.05, 32, tol, 1e-12, 1e-6) == pytest.approx(0.1 * tol)
    # block_m = 0 must never tighten
    assert _ns_target(0.05, 0, tol, 1e-14, 1e-6) == 0.05


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


def test_block_schedule_cuts_sweeps_on_goe():
    """Per-sweep block schedule: big blocks first inject the diagonal spread
    the early sweeps are bottlenecked on (the attempt-#9 anatomy). Must beat
    fixed m=32 by a clear margin on GOE, at full accuracy."""
    A = goe(400, seed=1)
    _, _, fixed = ssj_eigh(A, block_m=32, return_info=True)
    w, V, sched = ssj_eigh(A, block_m=[200, 100, 32], return_info=True)
    assert sched["converged"]
    assert sched["sweeps"] <= fixed["sweeps"] - 3
    assert_accurate(A, w, V)


def test_block_schedule_scalar_singleton_equivalent():
    """block_m=[32] must be exactly block_m=32 -- same path, same result."""
    A = goe(150, seed=6)
    w0, V0, i0 = ssj_eigh(A, block_m=32, return_info=True)
    w1, V1, i1 = ssj_eigh(A, block_m=[32], return_info=True)
    assert i0["sweeps"] == i1["sweeps"]
    assert np.array_equal(w0, w1)
    assert np.array_equal(V0, V1)


def test_block_schedule_survives_degeneracy():
    """Exact ties are the spectrum a schedule does NOT help (its bottleneck is
    tie resolution, not spread) -- it must still be correct there."""
    A = degenerate(200)
    w, V, info = ssj_eigh(A, block_m=[100, 50, 32], return_info=True)
    assert info["converged"]
    assert_accurate(A, w, V, tol=1e-11)


def test_block_schedule_entries_capped():
    """Schedule entries above n//2 are capped, like the scalar."""
    A = goe(60, seed=9)
    w, V, info = ssj_eigh(A, block_m=[500, 32], return_info=True)
    assert info["converged"]
    assert_accurate(A, w, V)


def test_mixed_schedule_full_phase_resolves_fp32_invisible_cluster():
    """precision="mixed" with a block schedule: the fp64 phase must keep the
    schedule's small tail blocks. A 1e-9 cluster is invisible at fp32
    resolution (~1e-7), so it reaches the fp64 phase unresolved -- with the
    tail blocks it costs 2 sweeps there, without them 5."""
    A = clustered(400, gap=1e-9)
    w, V, info = ssj_eigh(A, precision="mixed", block_m=[200, 100, 32],
                          return_info=True)
    assert info["converged"]
    assert info["sweeps"] <= 3          # fp64 phase alone
    assert_accurate(A, w, V, tol=1e-11)
