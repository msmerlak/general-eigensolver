"""How well does the one-hop screen rho_j actually predict convergence?

This is the evidence behind `_auto_gate` in ssj/dispatch.py. Collect
(rho_j, converged_j) over many columns and several matrix families, then ask
whether any threshold separates them.

The answer is no, and not by a little: the classes overlap across more than
three orders of magnitude in rho. What follows from that is not "the screen
is useless" but "the gate should be set by the COST of being wrong, not by
the accuracy of the screen" -- which is why the default differs between the
dense and sparse regimes. Run with `python bench_screen.py`.
"""
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, 'src')
sys.path.insert(0, '.')
from bench_sparse import sparse_diagonally_dominant, sparse_nonsymmetric  # noqa
from ssj import ipt_eig_partial  # noqa
from ssj.ipt import ipt_rate_columns  # noqa

rows = []


def collect(A, cols, tag, herm=True):
    rho = ipt_rate_columns(A, cols)
    _, _, info = ipt_eig_partial(A, cols, return_info=True, hermitian=herm,
                                 max_iter=400)
    ok = np.asarray(info["converged_cols"], dtype=bool)
    for r, c in zip(rho, ok):
        rows.append((float(r), bool(c), tag))


rng = np.random.default_rng(0)

# family 1: sparse diagonally dominant at a range of couplings
for seed in range(4):
    A0, d = sparse_diagonally_dominant(2000, seed=seed)
    W = A0 - sp.diags(d)
    order = np.argsort(np.abs(d - np.median(d)))
    cols = list(order[:12])
    for c in (1.0, 10.0, 40.0, 80.0, 120.0, 200.0, 400.0):
        collect(sp.diags(d) + c * W, cols, f"sparse/c={c}")

# family 2: dense near-diagonal at a range of couplings
for seed in range(3):
    r2 = np.random.default_rng(100 + seed)
    n = 400
    d = np.sort(r2.uniform(0, 100, n))
    M = r2.standard_normal((n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    for c in (0.01, 0.05, 0.1, 0.3, 1.0, 3.0):
        A = np.diag(d) + c * M
        collect(A, list(range(0, n, 40)), f"dense/c={c}")

# family 3: graded spectrum (small gaps at one end)
for seed in range(2):
    r3 = np.random.default_rng(200 + seed)
    n = 400
    d = np.cumsum(np.abs(r3.standard_normal(n)) ** 2)
    M = r3.standard_normal((n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    for c in (0.05, 0.5, 2.0):
        collect(np.diag(d) + c * M, list(range(0, n, 40)), f"graded/c={c}")

rho = np.array([r[0] for r in rows])
ok = np.array([r[1] for r in rows])
print(f"{len(rows)} columns, {ok.sum()} converged, {(~ok).sum()} diverged\n")

print("outcome by rho decile:")
print(f'{"rho range":>24} {"n":>5} {"converged":>10} {"rate":>7}')
edges = np.quantile(rho, np.linspace(0, 1, 11))
for i in range(10):
    lo, hi = edges[i], edges[i + 1]
    m = (rho >= lo) & (rho <= hi if i == 9 else rho < hi)
    if m.sum():
        print(f"{lo:11.4f} - {hi:8.4f} {m.sum():5} {ok[m].sum():10} "
              f"{ok[m].mean():7.2f}")

print("\nthreshold sweep (gate: try IPT when rho < gate):")
print(f'{"gate":>8} {"tried":>6} {"wasted":>7} {"missed":>7} {"note":>28}')
for g in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf):
    tried = rho < g
    wasted = int((tried & ~ok).sum())      # ran IPT, it diverged
    missed = int((~tried & ok).sum())      # would have worked, not tried
    note = ""
    if g == 0.1:
        note = "<- dense default"
    if g == np.inf:
        note = "<- sparse default"
    print(f"{g:8} {int(tried.sum()):6} {wasted:7} {missed:7} {note:>28}")

print("\nseparability: max rho among converged, min rho among diverged")
print(f"  converged: max rho = {rho[ok].max():.4f}")
print(f"  diverged : min rho = {rho[~ok].min():.4f}")
print("  -> overlapping" if rho[~ok].min() < rho[ok].max() else "  -> separable")
