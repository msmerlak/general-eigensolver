"""Acceleration explorations beyond mixed precision: over-relaxation, momentum
in generator space, and an unshifted-QR-algorithm prologue. Sweep counts are
deterministic, so these results are free of the machine's timing noise.

Run: python3 experiments_accel.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from ssj import off_frobenius  # noqa: E402
from ssj.core import _angles, _orth_qr  # noqa: E402

TOL = 1e-13
MAX_SWEEPS = 200


def goe(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2.0 * n)


def graded(n, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * 2.0 ** (-np.arange(n, dtype=float))) @ Q.T
    return (A + A.T) / 2.0


def run(A, step, X=None):
    """Iterate X <- step(X, K) until off(B) converges; return sweep count
    (MAX_SWEEPS+ marks failure) and the final off(B)/||A||."""
    n = A.shape[0]
    normA = np.linalg.norm(A, ord=2)
    X = np.eye(n) if X is None else X
    for sweep in range(MAX_SWEEPS):
        B = X.T @ (A @ X)
        B = (B + B.T) / 2.0
        rel = off_frobenius(B) / normA
        if not np.isfinite(rel) or rel > 1e6:
            return MAX_SWEEPS + 1, rel
        if rel <= TOL:
            return sweep, rel
        K = _angles(B)
        X = step(X, K)
    return MAX_SWEEPS + 1, rel


def qr_prologue(A, k):
    """k steps of the unshifted QR algorithm: B <- R Q, X accumulates Q."""
    n = A.shape[0]
    X = np.eye(n)
    B = A.copy()
    for _ in range(k):
        Q, R = np.linalg.qr(B)
        B = R @ Q
        X = X @ Q
    return X


if __name__ == "__main__":
    n = 200
    eye = np.eye(n)
    A = goe(n, seed=3)
    print(f"GOE N={n}, reference SSJ:", run(A, lambda X, K: _orth_qr(X @ (eye + K)))[0], "sweeps\n")

    print("## Over-relaxation X <- orth(X(I + gamma K))")
    for g in [1.1, 1.25, 1.5, 2.0]:
        s, rel = run(A, lambda X, K, g=g: _orth_qr(X @ (eye + g * K)))
        out = f"{s} sweeps" if s <= MAX_SWEEPS else f"FAILS (off = {rel:.2g})"
        print(f"  gamma = {g:4.2f}: {out}")

    print("\n## Momentum in generator space (transported): K_eff = K + beta * P")
    for beta in [0.2, 0.3, 0.5]:
        state = {"P": None}
        def step(X, K, beta=beta, state=state):
            Keff = K if state["P"] is None else K + beta * state["P"]
            Xn = _orth_qr(X @ (eye + Keff))
            Q = X.T @ Xn  # frame transport for the next sweep
            state["P"] = Q.T @ Keff @ Q
            return Xn
        s, rel = run(A, step)
        out = f"{s} sweeps" if s <= MAX_SWEEPS else f"FAILS (off = {rel:.2g})"
        print(f"  beta = {beta:3.1f}: {out}")

    print("\n## Unshifted-QR-algorithm prologue (k factorizations, then SSJ)")
    for name, M in [("GOE", A), ("graded 2^-i", graded(n, seed=3))]:
        base, _ = run(M, lambda X, K: _orth_qr(X @ (eye + K)))
        row = [f"{name}: 0 -> {base}"]
        for k in [3, 10, 30]:
            s, _ = run(M, lambda X, K: _orth_qr(X @ (eye + K)), X=qr_prologue(M, k))
            row.append(f"{k} -> {s}")
        print("  sweeps after k prologue steps:  " + ",  ".join(row))
