"""Generalized IPT: an arbitrary splitting A = M + R."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import ipt_eig_partial  # noqa: E402
from ssj.gipt import gipt_eig  # noqa: E402


def band_plus_dense(n, band, dense, seed=0):
    rng = np.random.default_rng(seed)
    d = np.arange(n, dtype=float)
    A = np.diag(d)
    i = np.arange(n - 1)
    A[i, i + 1] = band
    A[i + 1, i] = band
    M = A.copy()
    P = rng.standard_normal((n, n))
    P = (P + P.T) / 2.0
    np.fill_diagonal(P, 0.0)
    A = A + dense * P / np.max(np.abs(P))
    return A, M


def test_reduces_to_plain_ipt_when_M_is_the_diagonal():
    A, _ = band_plus_dense(200, 0.2, 0.02)
    M = np.diag(np.diag(A))
    lam, v, info = gipt_eig(A, M, 100, mode="reduced", return_info=True)
    w_plain, _ = ipt_eig_partial(A, [100])
    assert info["converged"]
    assert abs(lam - w_plain[0]) < 1e-10 * np.linalg.norm(A, 2)


def test_band_splitting_extends_the_basin():
    """The headline: M = band converges where M = diag diverges."""
    n, tgt = 400, 200
    A, M = band_plus_dense(n, 2.0, 0.02)
    assert not ipt_eig_partial(A, [tgt], return_info=True,
                               hermitian=True)[2]["converged"]
    lam, v, info = gipt_eig(A, M, tgt, mode="inverse", return_info=True)
    assert info["converged"]
    exact = np.linalg.eigvalsh(A)
    scale = np.linalg.norm(A, 2)
    assert np.min(np.abs(exact - lam)) / scale < 1e-11
    assert np.linalg.norm(A @ v - lam * v) / scale < 1e-10


def test_band_splitting_is_faster_where_both_work():
    n, tgt = 400, 200
    A, M = band_plus_dense(n, 0.5, 0.02)
    plain = ipt_eig_partial(A, [tgt], return_info=True, hermitian=True)[2]
    _, _, gen = gipt_eig(A, M, tgt, mode="inverse", return_info=True)
    assert plain["converged"] and gen["converged"]
    assert gen["iters"] < plain["iters"]


def test_inverse_mode_extends_the_basin_far():
    """M = band with inverse mode reaches band couplings where plain IPT and
    the strict IPT generalization both diverge."""
    n, tgt = 300, 150
    for band in (2.0, 8.0, 30.0):
        A, M = band_plus_dense(n, band, 0.02)
        assert not ipt_eig_partial(A, [tgt], return_info=True,
                                   hermitian=True)[2]["converged"]
        assert not gipt_eig(A, M, tgt, mode="reduced",
                            return_info=True)[2]["converged"]
        lam, v, info = gipt_eig(A, M, tgt, mode="inverse", return_info=True)
        assert info["converged"]
        exact = np.linalg.eigvalsh(A)
        scale = np.linalg.norm(A, 2)
        assert np.min(np.abs(exact - lam)) / scale < 1e-11


def test_inverse_mode_degenerates_with_a_diagonal_M():
    """Documented failure: with M = diag the single near-zero entry collapses
    the iterate onto e_target, so inverse mode must NOT be used there."""
    n, tgt = 300, 150
    A, _ = band_plus_dense(n, 2.0, 0.02)
    _, _, info = gipt_eig(A, np.diag(np.diag(A)), tgt, mode="inverse",
                          return_info=True)
    assert not info["converged"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
