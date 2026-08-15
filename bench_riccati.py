"""Head-to-head over many random instances: is the Brillouin-Wigner map a
strictly better drop-in for IPT, or does it just win on the cases I picked?

This is the evidence behind ssj/riccati.py. Run with `python bench_riccati.py`.

Same target, same tolerance, same starting point, cost counted in matvecs.
The question has three parts and the third is the one that decides shipping:
  1. does BW solve everything IPT solves? (no regression)
  2. does BW solve things IPT does not?   (robustness gain)
  3. when both solve, is BW cheaper?      (rate gain, or at least not worse)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ssj import ipt_eig_partial  # noqa: E402
from ssj.riccati import bw_eig_partial  # noqa: E402


def make(kind, n, coupling, seed):
    rng = np.random.default_rng(seed)
    d = np.sort(rng.uniform(0, 100, n))
    M = rng.standard_normal((n, n))
    if kind != "nonsym":
        M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    M *= coupling * (100.0 / n) / np.max(np.abs(M))
    A = np.diag(d) + M
    if kind == "degenerate":
        dd = np.diag(A).copy()
        j = n // 2
        dd[j + 1] = dd[j] + 10.0 ** rng.uniform(-7, -1)
        np.fill_diagonal(A, dd)
    if kind == "graded":                     # spectrum with a dense end
        dd = np.cumsum(np.abs(rng.standard_normal(n)) ** 2)
        np.fill_diagonal(A, dd)
    return A


KINDS = ["sym", "degenerate", "graded", "nonsym"]
COUPLINGS = [0.5, 1.0, 2.0, 4.0, 8.0]
N = 200
rows = []
for kind in KINDS:
    for cpl in COUPLINGS:
        for seed in range(6):
            A = make(kind, N, cpl, seed)
            j = N // 2
            sym = np.allclose(A, A.T)
            ev = np.linalg.eigvalsh(A) if sym else np.linalg.eigvals(A)
            scale = np.linalg.norm(A, 2)

            def solved(w, V, info):
                return bool(
                    info["converged"]
                    and np.min(np.abs(ev - w[0])) / scale < 1e-10
                    and np.linalg.norm(A @ V[:, 0] - w[0] * V[:, 0]) / scale
                    < 1e-9)

            a = ipt_eig_partial(A, [j], return_info=True, hermitian=sym,
                                max_iter=400)
            b = bw_eig_partial(A, [j], return_info=True, hermitian=sym,
                               max_iter=400)
            rows.append((kind, cpl, seed, solved(*a), a[2]["iters"],
                         solved(*b), b[2]["iters"]))

ipt_ok = sum(r[3] for r in rows)
bw_ok = sum(r[5] for r in rows)
both = [r for r in rows if r[3] and r[5]]
only_bw = [r for r in rows if r[5] and not r[3]]
only_ipt = [r for r in rows if r[3] and not r[5]]

print(f"{len(rows)} instances ({len(KINDS)} families x {len(COUPLINGS)} "
      f"couplings x 6 seeds), n={N}\n")
print(f"  IPT solved : {ipt_ok:3}/{len(rows)}")
print(f"  BW  solved : {bw_ok:3}/{len(rows)}")
print(f"  both       : {len(both):3}")
print(f"  union      : {ipt_ok + len(only_bw):3}   <- try IPT on BW's failures")
print(f"  BW only    : {len(only_bw):3}   <- robustness gained")
print(f"  IPT only   : {len(only_ipt):3}   <- BW is NOT a strict superset")

if both:
    ra = np.array([r[4] for r in both], float)
    rb = np.array([r[6] for r in both], float)
    print(f"\nwhere BOTH solve ({len(both)} cases), iterations:")
    print(f"  IPT median {np.median(ra):6.1f}   BW median {np.median(rb):6.1f}")
    print(f"  BW cheaper in {int(np.sum(rb < ra))}, equal {int(np.sum(rb == ra))},"
          f" dearer in {int(np.sum(rb > ra))}")
    print(f"  median ratio BW/IPT = {np.median(rb / ra):.3f}")

print("\nby family:")
print(f'{"family":12}{"IPT":>6}{"BW":>6}{"of":>6}')
for kind in KINDS:
    sub = [r for r in rows if r[0] == kind]
    print(f"{kind:12}{sum(r[3] for r in sub):6}{sum(r[5] for r in sub):6}"
          f"{len(sub):6}")

if only_ipt:
    print("\nregression cases (IPT solved, BW did not):")
    for r in only_ipt:
        print(f"  {r[0]:12} coupling={r[1]:<5} seed={r[2]}")
