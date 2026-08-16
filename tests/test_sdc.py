"""Tests for spectral divide and conquer.

The point of these is the case the rest of the repository cannot touch:
dense, non-normal, far-from-diagonal matrices. SSJ, IPT and the shears all
either stall or leave their basin there (see GENERAL.md); SDC does not.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj.sdc import matrix_sign, sdc_eigvals  # noqa: E402


def match_err(a, b, scale):
    b = list(b)
    worst = 0.0
    for x in a:
        d = [abs(x - y) for y in b]
        k = int(np.argmin(d))
        worst = max(worst, d[k])
        b.pop(k)
    return worst / scale


def test_matrix_sign_is_an_involution_with_right_rank():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((60, 60))
    S, iters = matrix_sign(A)
    assert np.linalg.norm(S @ S - np.eye(60), "fro") / 60 < 1e-12
    # trace of the spectral projector counts eigenvalues with Re > 0
    r = np.trace(0.5 * (np.eye(60) + S))
    assert abs(r - np.sum(np.linalg.eigvals(A).real > 0)) < 1e-6
    assert iters < 40


def test_sdc_on_ginibre():
    """Dense non-normal, far from diagonal: IPT diverges and shears plateau
    here, so this is the case SDC exists for."""
    rng = np.random.default_rng(1)
    for n in (60, 150):
        A = rng.standard_normal((n, n))
        w = sdc_eigvals(A)
        assert len(w) == n
        assert match_err(w, np.linalg.eigvals(A), np.linalg.norm(A, 2)) < 1e-10


def test_sdc_handles_complex_pairs():
    """Real matrix with a purely complex spectrum (block rotations)."""
    n = 40
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    B = np.zeros((n, n))
    for i in range(0, n, 2):
        a, b = rng.standard_normal(2)
        B[i, i] = B[i + 1, i + 1] = a
        B[i, i + 1], B[i + 1, i] = b, -b
    A = Q @ B @ Q.T
    w = sdc_eigvals(A)
    assert np.sum(np.abs(w.imag) > 1e-8) > 0
    assert match_err(w, np.linalg.eigvals(A), np.linalg.norm(A, 2)) < 1e-9


def test_sdc_symmetric():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((50, 50))
    A = (M + M.T) / 2.0
    w = sdc_eigvals(A)
    assert match_err(w, np.linalg.eigvals(A), np.linalg.norm(A, 2)) < 1e-12


def test_sdc_ill_conditioned_eigenproblem_is_backward_stable():
    """A random upper-triangular matrix has eigenvector condition ~1e19, so
    its eigenvalues are hypersensitive and NO backward-stable method can
    return them to high forward accuracy. What must hold is that SDC's answer
    is the exact spectrum of a nearby matrix -- i.e. it degrades like the
    conditioning, not worse, and does not silently produce garbage."""
    rng = np.random.default_rng(3)
    A = np.triu(rng.standard_normal((50, 50)))
    cond = np.linalg.cond(np.linalg.eig(A)[1])
    assert cond > 1e12, "test matrix is supposed to be ill-conditioned"
    w = sdc_eigvals(A)
    # forward error bounded by eps * conditioning, generously
    err = match_err(w, np.linalg.eigvals(A), np.linalg.norm(A, 2))
    assert err < 1e-1
    # and the multiset is still the right size and finite
    assert len(w) == 50 and np.all(np.isfinite(w))


def test_sdc_multiple_eigenvalue_falls_back_not_loops():
    """A block that cannot be split (repeated eigenvalue) must terminate."""
    A = np.eye(20) * 3.0
    A[0, 5] = 1.0  # defective-ish, still unsplittable by any vertical line
    w = sdc_eigvals(A)
    assert len(w) == 20
    assert np.allclose(np.sort(w.real), 3.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


def test_sdc_leaf_default_is_one_split():
    """The leaf default is max(2, n//2) -- one split, both halves to dgeev.
    Recursing to 2x2 measured 4x slower with no accuracy gain (SSJ_LOG #25),
    the same lesson the symmetric solver learned in #17. Pin the default and
    the accuracy so a future 'deeper must be better' change has to face it."""
    import numpy as np
    from ssj.sdc import sdc_eigvals

    r = np.random.default_rng(2)
    A = r.standard_normal((200, 200)) / np.sqrt(200)
    nrm = np.linalg.norm(A, 2)
    wref = np.sort_complex(np.linalg.eigvals(A))
    w = np.sort_complex(sdc_eigvals(A))
    assert np.max(np.abs(w - wref)) / nrm < 1e-10
    # the default must equal the explicit one-split configuration
    w2 = np.sort_complex(sdc_eigvals(A, min_block=100))
    assert np.max(np.abs(w - w2)) / nrm < 1e-12
