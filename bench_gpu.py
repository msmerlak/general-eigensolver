"""GPU benchmark for SSJ vs cuSOLVER, via CuPy.

Run on a machine with a CUDA GPU and cupy installed:

    python3 bench_gpu.py [N ...]        (default: 2000 4000 8000)

Measures, per size:
  - one FP64 gemm (the unit everything else is priced in)
  - cupy.linalg.eigh (cuSOLVER syevd) -- the incumbent
  - SSJ cold start, method="gemm" (matmul-only) and method="cholqr2"
  - SSJ warm start from the eigenbasis of a nearby matrix (eps = 1e-4),
    the tracking scenario -- against re-running cuSOLVER on the new matrix

The interesting outputs are the ratios: how many gemm-equivalents cuSOLVER
costs on your GPU (CPU LAPACK costs ~4-13, which is why SSJ loses there), and
whether SSJ's measured gemm count (~300-500 cold, ~10-30 warm per tracking
update at tol 1e-13) comes in under it.
"""
from __future__ import annotations

import sys
import time

try:
    import cupy as cp
except ImportError:
    sys.exit("cupy not installed -- run this on a CUDA machine "
             "(pip install cupy-cuda12x)")

sys.path.insert(0, "src")
from ssj import ssj_eigh  # noqa: E402


def sync_time(fn, repeat=3):
    fn()  # warmup (compilation, workspace allocation)
    cp.cuda.Stream.null.synchronize()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def bench(n):
    rng = cp.random.default_rng(0)
    M = rng.standard_normal((n, n))
    A = (M + M.T) / cp.sqrt(2.0 * n)

    t_gemm = sync_time(lambda: A @ A)
    t_eigh = sync_time(lambda: cp.linalg.eigh(A), repeat=1)
    print(f"\nN={n}:  gemm {t_gemm*1e3:8.1f} ms   "
          f"cuSOLVER eigh {t_eigh:7.3f} s  = {t_eigh/t_gemm:6.0f} gemm-equivalents")

    for method, kw in [("gemm", {}), ("cholqr2", {}),
                       ("gemm", {"precision": "mixed"})]:
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        w, V, info = ssj_eigh(A, method=method, return_info=True, **kw)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        label = method + (" mixed" if kw else "")
        g = ""
        if method == "gemm":
            g = f", {info['gemms']} f64 gemms"
            if "gemms_low" in info:
                g += f" + {info['gemms_low']} f32 gemms"
        print(f"  SSJ cold  {label:14s}: {info['sweeps']:3d} sweeps, {dt:7.3f} s "
              f"= {dt/t_gemm:6.0f} gemm-equivalents{g}  "
              f"({'converged' if info['converged'] else 'NOT CONVERGED'})")

    # tracking scenario
    _, V0 = cp.linalg.eigh(A)
    P = rng.standard_normal((n, n))
    P = (P + P.T) / 2.0
    A2 = A + 1e-4 * P / float(cp.linalg.norm(P, ord=2))
    t_eigh2 = sync_time(lambda: cp.linalg.eigh(A2), repeat=1)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    w, V, info = ssj_eigh(A2, X0=V0, method="gemm", return_info=True)
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0
    print(f"  SSJ warm (eps=1e-4, gemm): {info['sweeps']} sweeps, {info['gemms']} raw gemms, "
          f"{dt:.3f} s  vs cuSOLVER re-solve {t_eigh2:.3f} s  -> {t_eigh2/dt:.1f}x")

    # accuracy spot check
    resid = float(cp.linalg.norm(A2 @ V - V * w) / cp.linalg.norm(A2, ord="fro"))
    print(f"  warm-solve resid (vs ||A||_F): {resid:.1e}")


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [2000, 4000, 8000]
    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    print(f"GPU: {dev['name'].decode()}  (FP64 path; tol 1e-13)")
    for n in sizes:
        bench(n)
