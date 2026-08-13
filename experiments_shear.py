"""Norm-reducing shears for the general problem: the invariant that GENERAL.md
says is required.

The three failures in GENERAL.md all reduce to the same thing: for a general
matrix there is no orthogonally-decreasing off-diagonal quantity to descend.
Eberlein's answer is to descend a different invariant -- the departure from
normality

    Delta(A) = ||A||_F^2 - sum_i |lambda_i|^2  >= 0,   zero iff A is normal

which non-orthogonal (shear) similarities *can* decrease, since ||A||_F is not
similarity-invariant while the spectrum is.

The simultaneous (SSJ-style, all-pairs-at-once) form falls out in closed form.
For A <- T^{-1} A T with T = I + G, to first order A <- A + [A, G]. Split
G = K + S into antisymmetric K (rotation; preserves ||A||_F) and symmetric S
(shear). Then

    d||A||_F^2 = 2 <A, [A,S]> = 2 tr((A^T A - A A^T) S) = 2 <C, S>

with C = A^T A - A A^T the self-commutator: symmetric, traceless, and zero
exactly when A is normal. So steepest descent on the departure from normality
is simply

    S = -mu C

and tracelessness means the shear is volume-preserving, with no scaling drift
to control. The natural algorithm alternates:

  1. shear   A <- T^{-1} A T,  T = I - mu C     (drives A toward normal)
  2. rotate  SSJ-style orthogonal sweeps        (valid once A is normal:
             for normal A, minimizing off^2 = maximizing sum |a_ii|^2, the
             same structure as the symmetric problem)

Both steps need the SSJ lesson applied -- the step must be capped, here so
that T stays well-conditioned.

Run: python3 experiments_shear.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from ssj.core import _angles, _orth_qr  # noqa: E402


def departure_from_normality(A):
    """||A||_F^2 - sum |lambda|^2, normalized by ||A||_F^2."""
    fro2 = float(np.sum(np.abs(A) ** 2))
    lam2 = float(np.sum(np.abs(np.linalg.eigvals(A)) ** 2))
    return (fro2 - lam2) / max(fro2, 1e-300)


def commutator_norm(A):
    """||A^T A - A A^T||_F / ||A||_F^2 -- normality defect, no eigensolve."""
    C = A.conj().T @ A - A @ A.conj().T
    return float(np.linalg.norm(C, "fro")) / max(float(np.sum(np.abs(A) ** 2)), 1e-300)


def off_norm(A):
    return float(np.linalg.norm(A - np.diag(np.diag(A)), "fro"))


def shear_step(A, cap=0.25, line_search=True):
    """One simultaneous shear: A <- T^{-1} A T with T = I - mu C, C the
    self-commutator, mu set so ||mu C||_2 = cap (keeps T well-conditioned).

    Costs 2 gemms for C, one gemm for A@T, and one LU solve (N^3/3, i.e. about
    a sixth of a gemm) -- cheaper than a single SSJ sweep.

    The first-order descent guarantee only holds for small steps, so a fixed
    cap overshoots badly on matrices that are already near-normal (measured:
    a near-diagonal matrix starting at departure 0.000 was pushed to 0.028 by
    one cap=0.25 step). Backtracking until ||A||_F actually decreases is the
    saturation this step needs -- the same lesson as SSJ's bounded angles,
    and each trial costs only a solve plus a gemm.
    """
    n = A.shape[0]
    eye = np.eye(n)
    C = A.conj().T @ A - A @ A.conj().T
    nC = float(np.linalg.norm(C, 2))
    if nC == 0.0:
        return A, 0.0
    f0 = float(np.sum(np.abs(A) ** 2))
    mu = cap / nC
    for _ in range(30 if line_search else 1):
        T = eye - mu * C
        try:
            An = np.linalg.solve(T, A @ T)
        except np.linalg.LinAlgError:  # pragma: no cover
            mu *= 0.5
            continue
        if not line_search:
            return An, nC
        f1 = float(np.sum(np.abs(An) ** 2))
        if np.isfinite(f1) and f1 < f0:
            return An, nC
        mu *= 0.5
    return A, nC  # no decrease available: already at a stationary point


def rotate_sweep(A):
    """One SSJ sweep applied to the symmetric part's angle map, as an
    orthogonal similarity (valid diagonalizer once A is normal)."""
    n = A.shape[0]
    Sym = (A + A.conj().T) / 2.0
    K = _angles(Sym)
    Q = _orth_qr(np.eye(n) + K)
    return Q.conj().T @ A @ Q


def _normal_matrix(n, seed=3):
    """A genuinely normal, nonsymmetric matrix: Q (D + skew) Q^T with D and the
    skew part sharing the eigenbasis. The rotation phase should diagonalize it
    (up to real Schur 2x2 blocks for the complex pairs) with no shear at all."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    blocks = np.zeros((n, n))
    for i in range(0, n - 1, 2):
        a, b = rng.standard_normal(2)
        blocks[i, i] = blocks[i + 1, i + 1] = a
        blocks[i, i + 1], blocks[i + 1, i] = b, -b
    if n % 2:
        blocks[-1, -1] = rng.standard_normal()
    return Q @ blocks @ Q.T


def ginibre(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, n)) / np.sqrt(n)


def near_diag_gen(n, ratio, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    return np.diag(np.arange(n, dtype=float)) + ratio * W / np.max(np.abs(W))


def run(A, n_shear=200, n_rot=60, cap=0.25, verbose=False):
    """Sequenced, not interleaved: shear to normality first, then rotate.

    Rotations are orthogonal similarities, so they leave ||A||_F, the spectrum
    and hence the departure from normality exactly invariant -- interleaving
    them with shears buys nothing and only obscures which phase is stuck.
    Once A is normal its Hermitian and skew parts commute and therefore share
    eigenvectors, which is what makes diagonalizing the Hermitian part (the
    ordinary SSJ angle map) the right rotation to apply.
    """
    A = A.copy()
    normA = np.linalg.norm(A, 2)
    hist = []
    for it in range(n_shear):
        dn = commutator_norm(A)
        hist.append(dn)
        if verbose and it % 25 == 0:
            print(f"    shear {it:3d}: normality defect {dn:.3e}")
        if dn < 1e-13:
            break
        A, _ = shear_step(A, cap=cap)
    dn_final = commutator_norm(A)
    if verbose:
        print(f"    after shears: normality defect {dn_final:.3e}, "
              f"off {off_norm(A)/normA:.3e}")
    for it in range(n_rot):
        off = off_norm(A) / normA
        if verbose and it % 20 == 0:
            print(f"    rot   {it:3d}: off {off:.3e}")
        if off < 1e-12:
            break
        A = rotate_sweep(A)
    return A, hist, dn_final


if __name__ == "__main__":
    print("## Does the shear alone drive a matrix toward normal?\n")
    for name, A in [("Ginibre N=100", ginibre(100)),
                    ("general near-diagonal rho=0.5, N=100",
                     near_diag_gen(100, 0.5))]:
        B = A.copy()
        d0 = departure_from_normality(B)
        seq = [d0]
        for _ in range(30):
            B, _ = shear_step(B, cap=0.25)
            seq.append(departure_from_normality(B))
        print(f"  {name}")
        print(f"    departure from normality: " +
              "  ".join(f"{v:.3f}" for v in seq[:8]) +
              f"  ...  {seq[-1]:.3e} after 30 shears")

    print("\n## Shear to normality, then rotate: does it diagonalize?\n")
    for name, A in [("general near-diagonal rho=0.5, N=60", near_diag_gen(60, 0.5)),
                    ("Ginibre N=60", ginibre(60, seed=1)),
                    ("normal by construction (Q D Q^T + skew), N=60",
                     _normal_matrix(60))]:
        print(f"  {name}")
        B, hist, dn_final = run(A, verbose=True)
        off = off_norm(B) / np.linalg.norm(A, 2)
        lam_true = np.sort_complex(np.linalg.eigvals(A))
        lam_got = np.sort_complex(np.linalg.eigvals(B))
        drift = np.max(np.abs(lam_true - lam_got)) / np.linalg.norm(A, 2)
        # a real matrix with complex eigenvalues cannot go below its real
        # Schur 2x2 blocks, so report how much of `off` those blocks explain
        n_complex = int(np.sum(np.abs(lam_true.imag) > 1e-10))
        print(f"    final: normality defect {dn_final:.2e}, off {off:.2e}, "
              f"spectrum drift {drift:.2e}, {n_complex} complex eigenvalues\n")
