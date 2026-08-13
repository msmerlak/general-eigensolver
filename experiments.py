"""Mechanism experiments: convergence trajectory, monotonicity, and controlled
confirmations of the negative results in RESULTS.md (each variant below removes
one of the two saturations and is predicted to diverge).

Run: python3 experiments.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from ssj import ssj_eigh, off_frobenius  # noqa: E402
from ssj.core import _angles, _orth_qr, _power_iter_norm_antisym  # noqa: E402


def goe(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2.0 * n)


def toeplitz_21(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A, 2.0)
    i = np.arange(n - 1)
    A[i, i + 1] = 1.0
    A[i + 1, i] = 1.0
    return A


def custom_iteration(A, step, sweeps=60, tol=1e-13):
    """Run X <- step(X, K, B) with the standard angle map; return the off(B)
    trajectory (relative to ||A||_2)."""
    n = A.shape[0]
    normA = np.linalg.norm(A, ord=2)
    X = np.eye(n)
    traj = []
    for _ in range(sweeps):
        B = X.T @ (A @ X)
        B = (B + B.T) / 2.0
        rel = off_frobenius(B) / normA
        traj.append(rel)
        if not np.isfinite(rel) or rel > 1e6 or rel <= tol:
            break
        K = _angles(B)
        X = step(X, K, B)
    return traj


def fmt(traj):
    return "  ".join(f"{v:.2g}" for v in traj)


if __name__ == "__main__":
    n, eye = 200, np.eye(200)

    print("## Convergence trajectory (GOE N=200, method=auto): off(B)/||A|| per sweep\n")
    _, _, info = ssj_eigh(goe(200), return_info=True)
    print(fmt(info["history"]), "\n")

    print("## Monotonicity\n")
    worst = -np.inf
    for seed in range(20):
        _, _, info = ssj_eigh(goe(100, seed=seed), return_info=True)
        h = np.array(info["history"])
        worst = max(worst, float(np.max(np.diff(h))))
    print(f"20 GOE seeds (N=100): worst single-sweep change of off(B)/||A|| = {worst:.2e}")
    _, _, info = ssj_eigh(toeplitz_21(200), return_info=True)
    h = np.array(info["history"])
    print(f"Toeplitz (2,1) N=200: worst single-sweep change = {np.max(np.diff(h)):.2e}"
          f" (converged in {info['sweeps']} sweeps)\n")

    print("## Negative results (variants that remove a saturation)\n")
    A = goe(200, seed=3)

    t = custom_iteration(A, lambda X, K, B: _orth_qr(X @ (eye + K + 0.5 * (K @ K))))
    print(f"second-order retraction X(I+K+K^2/2): off after {len(t)} sweeps = {t[-1]:.2g}")

    state = {"i": 0}
    def deferred(X, K, B):
        state["i"] += 1
        Y = X @ (eye + K)
        return _orth_qr(Y) if state["i"] % 2 == 0 else Y
    t = custom_iteration(A, deferred)
    print(f"deferred orthonormalization (QR every 2nd sweep): off after {len(t)} sweeps = {t[-1]:.2g}")

    def compensated(X, K, B):
        # scale K by sigma/arctan(sigma) so the top K-plane rotates by exactly
        # its intended angle -- pre-compensating the retraction's arctan.
        # Predicted to diverge: every other plane over-rotates by up to the
        # same factor, exactly the aggressiveness the saturation exists to kill.
        sigma = _power_iter_norm_antisym(K)
        s = sigma / np.arctan(sigma) if sigma > 0 else 1.0
        return _orth_qr(X @ (eye + s * K))
    t = custom_iteration(A, compensated)
    print(f"arctan-compensated step (K scaled by sigma/arctan(sigma)): "
          f"off after {len(t)} sweeps = {t[-1]:.2g}")

    t = custom_iteration(A, lambda X, K, B: _orth_qr(X @ (eye + K)))
    print(f"reference (plain SSJ, same matrix): off after {len(t)} sweeps = {t[-1]:.2g}")
