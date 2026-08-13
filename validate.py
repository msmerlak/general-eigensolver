"""Reproduce the convergence battery and scaling experiments from RESULTS.md.

Run: python3 validate.py            (battery + scaling, ~1 min)
     python3 validate.py --full     (also N=1600 scaling row and timing runs)
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")
from ssj import ssj_eigh  # noqa: E402

TOL = 1e-13


# ---------------------------------------------------------------- test matrices

def diag_plus_coupling(n, mult, seed=0):
    """Sorted uniform diagonal + symmetric coupling scaled to `mult` times the
    median level spacing."""
    rng = np.random.default_rng(seed)
    d = np.sort(rng.uniform(-1.0, 1.0, n))
    spacing = np.median(np.diff(d))
    C = rng.standard_normal((n, n))
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 0.0)
    C *= mult * spacing / np.std(C[np.triu_indices(n, 1)])
    return np.diag(d) + C


def goe(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2.0 * n)


def zero_diagonal(n, seed=0):
    A = goe(n, seed)
    np.fill_diagonal(A, 0.0)
    return A


def toeplitz_21(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A, 2.0)
    i = np.arange(n - 1)
    A[i, i + 1] = 1.0
    A[i + 1, i] = 1.0
    return A


def wilkinson_plus(m=10):
    """W_{2m+1}^+ (n = 21 for m = 10)."""
    n = 2 * m + 1
    A = toeplitz_21(n)
    np.fill_diagonal(A, np.abs(np.arange(n) - m).astype(float))
    return A


def graded(n, seed=0):
    """Spectrum 2^-i, i = 0..n-1, in a random orthogonal basis."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * 2.0 ** (-np.arange(n, dtype=float))) @ Q.T
    return (A + A.T) / 2.0


def degenerate(n=500, clusters=10, mult=5, seed=0):
    """`clusters` exact `mult`-fold degeneracies, remaining eigenvalues simple,
    all drawn from the same O(1) scale."""
    rng = np.random.default_rng(seed)
    vals = np.concatenate([
        np.repeat(rng.uniform(-1.0, 1.0, clusters), mult),
        rng.uniform(-1.0, 1.0, n - clusters * mult),
    ])
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * vals) @ Q.T
    return (A + A.T) / 2.0


# ---------------------------------------------------------------- harness

def run(name, A, method="auto", tol=TOL, **kw):
    t0 = time.perf_counter()
    w, V, info = ssj_eigh(A, tol=tol, method=method, return_info=True, **kw)
    dt = time.perf_counter() - t0
    n = A.shape[0]
    norm2 = np.linalg.norm(A, ord=2)
    wt = np.linalg.eigvalsh(A)
    dlam = np.max(np.abs(w - wt))
    resid = np.linalg.norm(A @ V - V * w, ord="fro") / norm2
    ortho = np.linalg.norm(V.conj().T @ V - np.eye(n), ord="fro")
    flag = "" if info["converged"] else "  ** NOT CONVERGED **"
    extra = f"  gemms={info['gemms']}" if method == "gemm" else ""
    print(f"{name:52s} sweeps={info['sweeps']:4d}  dlam={dlam:.1e}  "
          f"resid={resid:.1e}  ortho={ortho:.1e}  ({dt:6.3f}s){extra}{flag}")
    return info


if __name__ == "__main__":
    full = "--full" in sys.argv

    print("## Convergence battery (tol 1e-13, cold start X0 = I)\n")
    run("diagonal + coupling at 1x level spacing, N=200", diag_plus_coupling(200, 1.0))
    run("same, 5x", diag_plus_coupling(200, 5.0))
    run("same, 100x", diag_plus_coupling(200, 100.0))
    run("GOE, N=200", goe(200))
    run("GOE, method=gemm (factorization-free)", goe(200), method="gemm")
    run("zero diagonal, N=200", zero_diagonal(200))
    run("tridiagonal Toeplitz (2,1), N=200", toeplitz_21(200))
    run("Wilkinson W_21^+", wilkinson_plus(10))
    run("graded spectrum 2^-i, N=200", graded(200))
    run("ten exact 5-fold degeneracies, N=500", degenerate(500, 10, 5))
    print()
    print("## Complex Hermitian (same map, anti-Hermitian K)\n")
    rng = np.random.default_rng(7)
    M = rng.standard_normal((200, 200)) + 1j * rng.standard_normal((200, 200))
    run("GUE, N=200", (M + M.conj().T) / np.sqrt(4.0 * 200))
    run("GUE, N=200, gemm", (M + M.conj().T) / np.sqrt(4.0 * 200), method="gemm")

    print()
    print("## Sweep count vs N (GOE, method=auto)\n")
    sizes = [100, 200, 400, 800] + ([1600] if full else [])
    for n in sizes:
        run(f"GOE N={n}", goe(n, seed=n))

    if full:
        print()
        print("## Retraction wall time (GOE N=1000, same sweeps expected)\n")
        A = goe(1000, seed=1000)
        for m in ["qr", "auto", "cholqr2", "gemm"]:
            run(f"GOE N=1000 method={m}", A, method=m)
