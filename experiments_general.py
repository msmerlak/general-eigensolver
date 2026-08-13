"""Exploring the general (nonsymmetric) problem.

Experiment 1 -- saturated IPT. IPT's denominator (Lambda_j - d_i) is the
*linearized* gap, the direct analogue of the linearized Jacobi angle
B_ij/(d_j - d_i) that RESULTS.md records diverging at ~0.85x the level
spacing. SSJ's fix was to replace it with the exact 2x2 solve, which
saturates. The same fix exists here: for the 2x2 block

    [[d_i, W_ij], [W_ji, d_j]],   delta = (d_j - d_i)/2,   p = W_ij W_ji

the exact eigenvalue nearest d_j is m + sign(delta) sqrt(delta^2 + p), so the
exact denominator is

    lambda - d_i = sign(delta) (|delta| + sqrt(delta^2 + p))

which reduces to (d_j - d_i) when p = 0 and tends to sqrt(p) as the gap
closes -- bounding v_i by sqrt(W_ij/W_ji) instead of letting it blow up.
Structurally identical to Jacobi's t = sign(tau)/(|tau| + sqrt(1+tau^2)).

Experiment 2 -- can orthogonal SSJ reach real Schur form? RESULTS.md reports
this direction failing because off^2 is not a Lyapunov function there. Tested
here with the exact 2x2 triangularizing angle rather than a linearized one.

Run: python3 experiments_general.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from ssj.core import _orth_qr  # noqa: E402


def near_diag_gen(n, ratio, seed=0):
    rng = np.random.default_rng(seed)
    d = np.arange(n, dtype=float)
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    W *= ratio / np.max(np.abs(W))
    return np.diag(d) + W


def ipt_run(A, saturated, max_iter=300, tol=1e-13):
    """IPT with either the linearized or the saturated (exact 2x2) denominator.
    Returns (iters, converged, final error)."""
    n = A.shape[0]
    d = np.diag(A).astype(np.complex128).copy()
    W = (A - np.diag(np.diag(A))).astype(np.complex128)
    V = np.eye(n, dtype=np.complex128)
    idx = np.arange(n)
    norm = np.linalg.norm(A, "fro") / np.sqrt(n)
    err0 = None
    if saturated:
        # p_ij = W_ij W_ji is fixed across iterations; only Lambda moves.
        P = W * W.T
    for it in range(1, max_iter + 1):
        WV = W @ V
        Lam = d + np.diag(WV)
        if saturated:
            delta = (Lam[None, :] - d[:, None]) / 2.0
            root = np.sqrt(delta * delta + P)
            # pick the branch continuous with the linearized gap (root -> |delta|)
            sgn = np.where(np.real(delta) >= 0, 1.0, -1.0)
            den = sgn * (sgn * delta + root)
        else:
            den = Lam[None, :] - d[:, None]
        den[idx, idx] = 1.0
        Vn = WV / den
        Vn[idx, idx] = 1.0
        err = float(np.max(np.abs(Vn - V)))
        V = Vn
        if err0 is None:
            err0 = max(err, 1e-300)
        if err <= tol * norm:
            return it, True, err
        if not np.isfinite(err) or err > 1e3 * err0:
            return it, False, err
    return max_iter, False, err


def schur_angles(B):
    """Exact 2x2 triangularizing rotation angles for every pair (i>j).

    For the block [[a, b], [c, d]] (rows/cols j, i) the rotation that makes the
    lower entry vanish is theta = atan2 of the eigenvector; where the block has
    a complex pair no real rotation triangularizes it and the angle is zero
    (that pair is a 2x2 block of the real Schur form).
    """
    n = B.shape[0]
    d = np.diag(B)
    a = d[:, None] * np.ones((1, n))     # a_ij = d_i
    dd = np.ones((n, 1)) * d[None, :]    # d_ij = d_j
    b = B                                 # upper coupling B_ij
    c = B.T                               # lower coupling B_ji
    half = (a - dd) / 2.0
    disc = half * half + b * c
    real = disc >= 0
    root = np.sqrt(np.where(real, disc, 0.0))
    # eigenvalue nearest d_i, then the rotation aligning that eigenvector
    lam = (a + dd) / 2.0 + np.sign(half) * root
    theta = np.arctan2(np.where(real, c, 0.0), np.where(real, lam - dd, 1.0))
    theta = np.where(real, theta, 0.0)
    K = np.tril(theta, -1)
    return K - K.T


def ssj_schur(A, max_sweeps=200, tol=1e-12):
    """SSJ in the Schur direction: simultaneous exact 2x2 triangularizing
    rotations, retracted orthogonally. Target: ||tril(B,-1)||_F -> 0."""
    n = A.shape[0]
    X = np.eye(n)
    normA = np.linalg.norm(A, 2)
    traj = []
    for sweep in range(max_sweeps):
        B = X.T @ A @ X
        low = np.linalg.norm(np.tril(B, -1), "fro") / normA
        traj.append(low)
        if low <= tol or not np.isfinite(low) or low > 1e6:
            break
        K = schur_angles(B)
        X = _orth_qr(X @ (np.eye(n) + K))
    return traj


if __name__ == "__main__":
    print("## Experiment 1: does saturating IPT's denominator widen its basin?\n")
    print(f"{'rho':>8} {'linearized':>22} {'saturated (exact 2x2)':>24}")
    for ratio in [0.01, 0.1, 0.3, 0.5, 0.8, 1.2, 2.0]:
        A = near_diag_gen(300, ratio)
        i1, c1, e1 = ipt_run(A, saturated=False)
        i2, c2, e2 = ipt_run(A, saturated=True)
        f = lambda i, c, e: (f"{i} its" if c else f"diverged ({e:.1g})")
        print(f"{ratio:>8} {f(i1,c1,e1):>22} {f(i2,c2,e2):>24}")

    print("\n## Experiment 2: SSJ in the Schur direction (orthogonal only)\n")
    for name, A in [
        ("general near-diagonal, rho=0.1", near_diag_gen(100, 0.1)),
        ("general random (Ginibre), N=100",
         np.random.default_rng(0).standard_normal((100, 100)) / 10.0),
    ]:
        traj = ssj_schur(A)
        tail = "  ".join(f"{v:.2g}" for v in traj[:6])
        end = f"{traj[-1]:.2g}"
        verdict = ("converged" if traj[-1] <= 1e-12
                   else "STALLED/DIVERGED")
        print(f"  {name}")
        print(f"    ||tril||/||A||: {tail} ...  final {end} after {len(traj)} sweeps  -> {verdict}")
