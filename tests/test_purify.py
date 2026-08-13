"""Purification: fixed points are spectral projectors, not eigenvectors."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj.purify import purify, purify_split, spectral_projector  # noqa: E402


def goe(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2.0 * n)


def test_projector_is_idempotent_and_commutes():
    A = goe(200)
    ev = np.linalg.eigvalsh(A)
    mu = float(np.median(ev))
    P, iters = spectral_projector(A, mu)
    assert np.linalg.norm(P @ P - P, "fro") < 1e-10
    assert np.linalg.norm(A @ P - P @ A, "fro") < 1e-10
    # rank counts the eigenvalues below mu, exactly
    assert abs(np.trace(P).real - np.sum(ev < mu)) < 1e-6


def test_globally_convergent_from_any_split_point():
    """The [0,1] scaling is a guarantee, so no shift should fail -- including
    ones near the spectral edges where Newton-Schulz on the sign function
    would be outside its region."""
    A = goe(150, seed=2)
    ev = np.linalg.eigvalsh(A)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        mu = float(np.quantile(ev, q))
        P, iters = spectral_projector(A, mu)
        assert np.linalg.norm(P @ P - P, "fro") < 1e-9
        assert abs(np.trace(P).real - np.sum(ev < mu)) < 1e-6


def test_purify_uses_only_matrix_products():
    """No factorization may appear: the whole point is inverse-freedom."""
    A = goe(120, seed=3)
    mu = float(np.median(np.linalg.eigvalsh(A)))
    count = {}
    spectral_projector(A, mu, count=count)
    assert count["gemm"] > 0
    assert "inv" not in count and "qr" not in count


def test_split_block_triangularizes():
    A = goe(150, seed=4)
    ev = np.linalg.eigvalsh(A)
    mu = float(np.median(ev))
    out = purify_split(A, mu)
    assert out is not None
    A11, A22, r, resid = out
    assert resid < 1e-9
    assert r == int(np.sum(ev < mu))
    # the two blocks' spectra reconstruct the original
    got = np.sort(np.concatenate([np.linalg.eigvalsh(A11),
                                  np.linalg.eigvalsh(A22)]))
    assert np.max(np.abs(got - ev)) / np.linalg.norm(A, 2) < 1e-10


def test_idempotent_input_is_a_fixed_point():
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((80, 80)))
    P0 = Q[:, :30] @ Q[:, :30].T
    P, iters = purify(P0.copy())
    assert iters == 1                      # already a projector
    assert np.linalg.norm(P - P0, "fro") < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
