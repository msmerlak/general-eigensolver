"""The saturated-Jacobi generator _angles.

It is O(n^2) but measured a third of the whole sweep at n=400, so it is built
in place and skips two scans when the diagonal has no ties. These tests pin
the properties that made those shortcuts legal, against a direct transcription
of the formula rather than against the optimized code itself.
"""
import numpy as np
import pytest

from ssj.core import _angles


def reference(B):
    """Straight transcription of K_ij = 1/2 arctan(2|B_ij|/(d_j - d_i)),
    with no shortcuts. Deliberately naive: this is the oracle."""
    n = B.shape[0]
    d = np.real(np.diag(B))
    absB = np.abs(B)
    gap = d[None, :] - d[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = 0.5 * np.arctan(2.0 * absB / gap)
    theta = np.nan_to_num(theta, nan=0.0)
    tie = (gap == 0.0) & (absB > 0.0)
    if bool(tie.any()):
        one = np.ones((n, n), dtype=theta.dtype)
        theta = np.where(tie, (np.pi / 4.0) * (np.triu(one, 1)
                                               - np.tril(one, -1)), theta)
    if B.dtype.kind == "c":
        nz = absB > 0
        phase = np.where(nz, B / np.where(nz, absB, 1.0), 0.0)
    else:
        phase = np.sign(B)
    K = theta * phase
    np.fill_diagonal(K, 0.0)
    return K


def goe(n, seed=0):
    r = np.random.default_rng(seed)
    M = r.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2 * n)


def herm(n, seed=0):
    r = np.random.default_rng(seed)
    M = r.standard_normal((n, n)) + 1j * r.standard_normal((n, n))
    return (M + M.conj().T) / np.sqrt(2 * n)


def degenerate(n, fold=5, seed=2):
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.standard_normal((n, n)))
    v = np.repeat(r.standard_normal(n // fold + 1), fold)[:n]
    A = (Q * v) @ Q.T
    return (A + A.T) / 2


def _zero_diag(A):
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    return A


CASES = {
    "goe": goe(64, 1),
    "hermitian": herm(64, 1),
    "degenerate": degenerate(60),
    "zero diagonal": _zero_diag(goe(64, 4)),
    "zero diagonal complex": _zero_diag(herm(64, 4)),
    "identity": np.eye(32),
    "all zeros": np.zeros((16, 16)),
    "pure diagonal": np.diag(np.arange(24.0)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_reference_exactly(name):
    """Bit-for-bit, not approximately -- the in-place build reorders nothing
    numerically, and a tolerance here would hide it if it ever did."""
    B = CASES[name]
    assert np.array_equal(_angles(B.copy()), reference(B.copy()))


@pytest.mark.parametrize("name", sorted(CASES))
def test_is_anti_hermitian_and_finite(name):
    """Anti-Hermiticity is what makes orth(I+K) a rotation; the singular
    points (zero gap, zero coupling) are exactly where it could be lost."""
    K = _angles(CASES[name].copy())
    assert np.isfinite(K).all()
    assert np.abs(K + K.conj().T).max() == 0.0
    assert np.abs(np.diag(K)).max() == 0.0


def test_tie_detection_fires_on_a_single_tied_pair():
    """The O(n log n) diagonal check replaced an O(n^2) mask. One tied pair
    anywhere must still take the saturating branch."""
    A = goe(48, 8)
    d = np.diag(A).copy()
    d[3] = d[7]
    np.fill_diagonal(A, d)
    K = _angles(A)
    assert np.array_equal(K, reference(A))
    # the tied pair saturates at exactly pi/4 in magnitude
    assert np.abs(abs(K[3, 7]) - np.pi / 4) < 1e-15


def test_angles_bounded_by_quarter_pi():
    """The arctan saturation: no pair angle may exceed pi/4."""
    for B in CASES.values():
        assert np.abs(_angles(B.copy())).max() <= np.pi / 4 + 1e-15


def test_input_is_not_mutated():
    """_angles builds in place internally; it must not touch its argument."""
    for B in (goe(32, 3), herm(32, 3), degenerate(30)):
        before = B.copy()
        _angles(B)
        assert np.array_equal(B, before)
