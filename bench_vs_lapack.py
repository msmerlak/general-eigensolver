"""Head-to-head against LAPACK on the niches where this repository's methods
can actually win: near-diagonal input (IPT alone) and tracking a slowly
varying matrix (warm-started IPT). Everything is priced in gemm-equivalents
(one N-by-N-by-N dgemm = 1 unit) as well as wall clock, because this container
is shared and wall times carry ~30% noise while flop ratios do not.

Run: python3 bench_vs_lapack.py [N ...]     (default 1000 2000)
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")
from ssj import ipt_eigh, ssj_eigh, ssj_ipt_eigh  # noqa: E402

REPEAT = 3


def best(fn, repeat=REPEAT):
    out = None
    t = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        t = min(t, time.perf_counter() - t0)
    return t, out


def gemm_time(n, repeat=3):
    A = np.random.default_rng(0).standard_normal((n, n))
    t, _ = best(lambda: A @ A, repeat)
    return t


def near_diagonal(n, ratio, seed=0):
    """Unit level spacing, max |W_ij| = ratio. IPT's rate is ~ratio."""
    rng = np.random.default_rng(seed)
    d = np.arange(n, dtype=float)
    W = rng.standard_normal((n, n))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    W *= ratio / np.max(np.abs(W))
    return np.diag(d) + W


def goe(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2.0 * n)


def acc(A, w, V):
    n = A.shape[0]
    norm2 = np.linalg.norm(A, ord=2)
    dlam = np.max(np.abs(w - np.linalg.eigvalsh(A))) / norm2
    resid = np.linalg.norm(A @ V - V * w, "fro") / norm2
    ortho = np.linalg.norm(V.T @ V - np.eye(n), "fro")
    return dlam, resid, ortho


def bench_near_diagonal(n, tg):
    print(f"\n### Near-diagonal, N={n} (IPT alone vs LAPACK)\n")
    print(f"{'coupling/gap':>13} {'IPT its':>8} {'IPT':>10} {'LAPACK':>10} "
          f"{'speedup':>8}   {'rel dlam':>9} {'resid':>9} {'ortho':>9}")
    for ratio in [0.2, 0.05, 0.01, 0.002]:
        A = near_diagonal(n, ratio)
        tl, (wl, _) = best(lambda: np.linalg.eigh(A))
        ti, (w, V, info) = best(lambda: ipt_eigh(A, return_info=True))
        if not info["converged"]:
            print(f"{ratio:>13} {'--':>8}  diverged (outside IPT's basin)")
            continue
        d_, r_, o_ = acc(A, w, V)
        print(f"{ratio:>13} {info['iters']:>8} {ti:>8.3f}s {tl:>8.3f}s "
              f"{tl/ti:>7.2f}x   {d_:>9.1e} {r_:>9.1e} {o_:>9.1e}")
        print(f"{'':>13} {'':>8} {ti/tg:>7.1f}g {tl/tg:>7.1f}g")


def bench_tracking(n, tg):
    """Track a slowly varying matrix: A(t+dt) = A(t) + eps * P.

    Warm path: B = V^T A_new V (2 gemms, near-diagonal by construction), IPT
    on B (k gemms), compose V @ V_ipt (1 gemm). LAPACK must re-solve cold.
    """
    print(f"\n### Tracking, N={n} (warm IPT vs cold LAPACK re-solve)\n")
    A = goe(n, seed=n)
    _, V0 = np.linalg.eigh(A)
    rng = np.random.default_rng(7)
    P = rng.standard_normal((n, n))
    P = (P + P.T) / 2.0
    P /= np.linalg.norm(P, ord=2)

    print(f"{'step eps':>9} {'IPT its':>8} {'warm':>10} {'LAPACK':>10} "
          f"{'speedup':>8}   {'rel dlam':>9} {'resid':>9} {'ortho':>9}")
    for eps in [1e-2, 1e-4, 1e-6]:
        A2 = A + eps * P
        tl, _ = best(lambda: np.linalg.eigh(A2))

        def warm():
            B = V0.T @ (A2 @ V0)
            B = (B + B.T) / 2.0
            w, Vd, info = ipt_eigh(B, return_info=True)
            return w, V0 @ Vd, info

        tw, (w, V, info) = best(warm)
        if not info["converged"]:
            print(f"{eps:>9.0e} {'--':>8}  IPT did not converge in the warm frame")
            continue
        d_, r_, o_ = acc(A2, w, V)
        print(f"{eps:>9.0e} {info['iters']:>8} {tw:>8.3f}s {tl:>8.3f}s "
              f"{tl/tw:>7.2f}x   {d_:>9.1e} {r_:>9.1e} {o_:>9.1e}")
        print(f"{'':>9} {'':>8} {tw/tg:>7.1f}g {tl/tg:>7.1f}g"
              f"   (warm = 2 form + {info['iters']} IPT + 1 compose gemms)")


def bench_global(n, tg):
    """Hard input (GOE): nobody beats LAPACK here, but the hybrid should beat
    plain SSJ by replacing the quadratic tail with one-gemm iterations."""
    print(f"\n### Hard input (GOE), N={n}: hybrid vs plain SSJ vs LAPACK\n")
    A = goe(n, seed=n)
    tl, _ = best(lambda: np.linalg.eigh(A), 2)
    ts, (_, _, si) = best(lambda: ssj_eigh(A, method="gemm", precision="mixed",
                                           return_info=True), 2)
    th, (w, V, hi) = best(lambda: ssj_ipt_eigh(A, method="gemm",
                                               precision="mixed",
                                               return_info=True), 2)
    d_, r_, o_ = acc(A, w, V)
    print(f"  LAPACK eigh          {tl:7.3f}s ({tl/tg:5.1f}g)")
    print(f"  SSJ gemm+mixed       {ts:7.3f}s ({ts/tg:5.1f}g)  "
          f"{si['sweeps']} sweeps")
    print(f"  SSJ->IPT hybrid      {th:7.3f}s ({th/tg:5.1f}g)  "
          f"{hi['sweeps']} sweeps + {hi['ipt_iters']} IPT iters "
          f"({hi['ipt_gemms']} probe gemms), path={hi['path']}")
    print(f"  hybrid accuracy: rel dlam {d_:.1e}, resid {r_:.1e}, ortho {o_:.1e}")
    print(f"  hybrid vs plain SSJ: {ts/th:.2f}x     vs LAPACK: {tl/th:.2f}x")


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [1000, 2000]
    for n in sizes:
        tg = gemm_time(n)
        print(f"\n{'='*78}\nN={n}: one dgemm = {tg*1e3:.1f} ms "
              f"(the gemm-equivalent unit, 'g' below)\n{'='*78}")
        bench_near_diagonal(n, tg)
        bench_tracking(n, tg)
        bench_global(n, tg)
