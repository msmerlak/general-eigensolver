"""Zolotarev sign function: coefficient correctness, and the measured boundary
of where it actually helps."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj.sdc import matrix_sign  # noqa: E402
from ssj.zolo import (matrix_sign_zolo, partial_fractions,  # noqa: E402
                      zolotarev_coeffs, zolotarev_sign_scalar)


def test_equioscillation_signature():
    """The best rational approximant of type (2r+1, 2r) equioscillates 2r+1
    times -- the cheapest complete check that the elliptic-function
    coefficients use the right conventions."""
    for ell, r in [(1e-2, 4), (1e-3, 8)]:
        x = np.geomspace(ell, 1.0, 4000)
        err = zolotarev_sign_scalar(x, ell, r) - 1.0
        assert np.sum(np.diff(np.sign(err)) != 0) == 2 * r + 1


def test_accuracy_improves_with_degree():
    ell = 1e-3
    prev = np.inf
    for r in (1, 2, 4, 8):
        x = np.geomspace(ell, 1.0, 2000)
        e = np.max(np.abs(zolotarev_sign_scalar(x, ell, r) - 1.0))
        assert e < prev
        prev = e
    assert prev < 1e-3


def test_odd_symmetry():
    assert np.allclose(zolotarev_sign_scalar([-0.3, -0.9], 1e-3, 4),
                       -zolotarev_sign_scalar([0.3, 0.9], 1e-3, 4))


def test_partial_fractions_match_product_form():
    c, M = zolotarev_coeffs(1e-3, 6)
    poles, a = partial_fractions(c)
    t = np.geomspace(1e-6, 1.0, 500)
    prod = np.ones_like(t)
    r = len(c) // 2
    for j in range(1, r + 1):
        prod = prod * (t + c[2 * j - 1]) / (t + c[2 * j - 2])
    frac = 1.0 + sum(a[j] / (t + poles[j]) for j in range(r))
    assert np.max(np.abs(prod - frac)) < 1e-9


def test_two_passes_reach_double_precision():
    """The headline claim of the Zolotarev construction."""
    ell, r = 1e-3, 8
    x = np.geomspace(ell, 1.0, 3000)
    z1 = zolotarev_sign_scalar(x, ell, r)
    e1 = np.max(np.abs(z1 - 1.0))
    z2 = zolotarev_sign_scalar(z1, max(1.0 - e1, 1e-12), r)
    assert np.max(np.abs(z2 - 1.0)) < 1e-14


def test_matrix_sign_real_spectrum_beats_newton_on_iterations():
    """Where Zolotarev is designed to win: a real spectrum."""
    rng = np.random.default_rng(0)
    n = 120
    M = rng.standard_normal((n, n))
    A = (M + M.T) / 2.0
    A -= np.trace(A) / n * np.eye(n)
    S_n, it_n = matrix_sign(A)
    S_z, it_z = matrix_sign_zolo(A, r=8)
    assert it_z < it_n / 2
    for S in (S_n, S_z):
        assert np.linalg.norm(S @ S - np.eye(n), "fro") / n < 1e-12
    # Compare against GROUND TRUTH, not against Newton. On this matrix the
    # smallest eigenvalue sits at 1.4e-3 of the spectral radius -- right on
    # the splitting line -- and Newton, still converging slowly there,
    # misclassifies it (trace -2, i.e. 59 positive) while Zolotarev gets the
    # true count (trace -4, 58 positive). Accuracy near the line is the
    # second thing the best-approximation property buys, after speed.
    true_trace = float(np.sum(np.linalg.eigvals(A).real > 0) * 2 - n)
    assert abs(np.trace(S_z) - true_trace) < 1e-6


def test_matrix_sign_complex_spectrum_still_correct():
    """Zolotarev is NOT the right tool for a complex spectrum (it is optimal on
    a real interval), but it must still return a correct sign function rather
    than silently wrong output."""
    rng = np.random.default_rng(1)
    n = 100
    A = rng.standard_normal((n, n))
    A -= np.trace(A) / n * np.eye(n)
    S, _ = matrix_sign_zolo(A, r=8)
    assert np.linalg.norm(S @ S - np.eye(n), "fro") / n < 1e-11
    S_n, _ = matrix_sign(A)
    assert abs(np.trace(S) - np.trace(S_n)) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
