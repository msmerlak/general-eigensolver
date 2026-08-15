# general-eigensolver

A Python/NumPy implementation of **Simultaneous Saturated Jacobi (SSJ)**: full
symmetric / Hermitian eigendecomposition by applying *every* classical Jacobi
rotation angle at once through a single linearized step, followed by
reprojection onto the orthogonal manifold. Parameter-free — no pivoting, no
shifts, no deflation or degeneracy-detection logic; matrix multiplication and
an elementwise arctangent are the whole map.

```
X ← I
repeat
    B ← XᵀA X ;  d ← diag(B)
    if ‖offdiag(B)‖_F ≤ ε‖A‖₂ : break
    K_ij ← ½·atan( 2B_ij / (d_j − d_i) )      (±π/4 at zero gap; antisymmetric)
    X ← orth( X(I + K) )                       (QR, or Newton–Schulz endgame)
return sort(diag(B)), columns of X
```

The map is self-stabilizing through two automatic saturations: the arctan
bounds each pair angle by π/4, and the reprojection of I + K rotates each
K-invariant plane by arctan(σ) rather than σ, saturating the *composed* step.
See [ALGORITHM.md](ALGORITHM.md) for the full specification and
[RESULTS.md](RESULTS.md) for the original (Julia) measurements this
implementation reproduces. Our measurements, including two refinements found
during reimplementation and several confirmed negative results, are in
[BENCHMARKS.md](BENCHMARKS.md).

The repository also implements **IPT** (Iterative Perturbation Theory), a
one-gemm-per-iteration fixed point for near-diagonal input, and the
**SSJ→IPT hybrid** that uses SSJ's global basin to feed IPT's cheap endgame.
Where LAPACK is beaten, at full accuracy:

| problem | input | vs LAPACK | measured |
|---|---|---|---|
| symmetric | near-diagonal ($\rho \lesssim 0.03$) | **1.4–1.9×** faster than `dsyevd` | [BENCHMARKS.md](BENCHMARKS.md) |
| **general** | near-diagonal ($\rho \lesssim 0.1$) | **4–12×** faster than `dgeev` | [GENERAL.md](GENERAL.md) |
| **normal** (nonsymmetric, complex spectrum) | any | **1.3–1.9×** faster than `dgeev`, and *unitary* eigenvectors | [GENERAL.md](GENERAL.md) |

For **few eigenpairs**, `ipt_eig_partial(A, cols)` exploits IPT's
column-separability to compute k targeted eigenpairs at O(N²k) — **4–7.7×
faster than ARPACK with shift-invert** on interior targets, with no
factorization at all, and up to 143× cheaper than a full solve.

Crucially the **basin is per-column too** (`ipt_rate_columns`, O(Nk)), which
is much weaker than needing a near-diagonal matrix: a dense strongly-coupled
band with global ρ = 992, where the full solver diverges, still yields its
isolated impurity/defect levels to machine precision — **20–123× faster than
ARPACK**.

For arbitrary input use **`eig_partial(A, sigma=..., k=...)`**, which screens
each target and routes it to IPT or ARPACK. The per-column rate is a *one-hop
heuristic and is optimistic*, so the router is built so a wrong screen costs
time, never correctness — unconverged IPT output is discarded and the target
re-run on ARPACK. Measured: 6.4–9.1× on isolated targets, and 0.0% overhead
when nothing qualifies. See [GENERAL.md](GENERAL.md).

The screen is in fact a *weak classifier*: over 576 columns (`bench_screen.py`)
the highest ρ among converged targets is 22.9 and the lowest among divergent
ones is 0.0068, so the classes overlap across three orders of magnitude and no
threshold separates them. What makes the router sound is the **per-column
outcome**: `ipt_eig_partial` reports `converged_cols` / `failed`, columns
retire independently (one bad target no longer aborts its neighbours
mid-flight), and the fallback re-runs only the targets that actually failed.

It follows that the gate should be set by the *cost of being wrong*, not the
accuracy of the screen — and that cost differs by orders of magnitude between
regimes. `eig_partial` now picks it automatically and **accepts sparse input**:
dense keeps the conservative 0.1 (measured: raising it loses, 0.023 s → 0.111 s
at N=400), while sparse tries everything, because a fully wasted attempt costs
0.4% of the fallback there. With automatic routing on sparse interior targets:
**184× at N=2000, 1,766× at N=5000** versus ARPACK shift-invert.

**Largest margin in the repository — large sparse, interior targets.** On
sparse matrices with wide diagonal spread and weak coupling, interior targets
force Krylov methods into shift-invert, whose LU fill-in explodes on random
sparsity (88× → 431× of nnz as N goes 2k → 10k, infeasible at 20k). IPT needs
3–5 sparse matvecs and no factorization: **8× → 347× faster than the best
alternative** (LOBPCG on the shifted-squared operator; ~20,000× vs ARPACK
shift-invert), with the margin *growing* with N. Cost is genuinely O(nnz):
**N=200,000 with 3.4M nonzeros, four interior eigenpairs, 0.24 s.** The
**nonsymmetric** case is larger still — no LOBPCG equivalent exists there, so
shift-invert is the only alternative and its fill-in is worse: **382× → 13,234×**
at N=2k → 10k. See `bench_sparse.py`.

The envelope, measured (`python bench_sparse.py --envelope`): it is **not**
limited to a handful of eigenpairs — cost is linear in k at a flat ~6 ms per
eigenpair with the iteration count still 3–4 at k=1024 (1024 interior
eigenpairs of a 20,000-square matrix in 6 s). The binding constraint is
coupling, not k: ρ ≲ 0.05 converged on every instance tried, ρ ≳ 0.25
diverged on every one, and *in between the outcome is instance-dependent* —
two instances measured here cross, one converging at ρ = 0.122 while the
other diverges at ρ = 0.096. Cost also degrades well before correctness does
(3–6 iterations at ρ ≲ 0.02, 20+ near the edge), and divergence can be slow
enough that a tight `max_iter` misreports slow convergence as failure.

Past that coupling boundary, **`sparse_block_ipt_eig`** solves the block
fixed point without ever forming a submatrix (the dense path builds `A[C,C]`,
3.2 GB at N=20,000; putting the block identity and the tail in one n×b array
makes one full matvec yield both halves). It roughly quadruples the usable
band — converging at ρ = 0.38 where plain IPT diverges — and `eig_partial`
escalates to it automatically on just the targets that failed, when the size
makes it worth it: **26–54× at N=5000**, versus 0.27–1.3× at N=2000, because
a block attempt is linear in N while the shift-invert fallback is not. Three
counterintuitive details (a *larger* block cap is worse; block IPT is 150×
slower than plain IPT when plain IPT would have worked; failure must be made
cheap or escalating is a net loss) are measured in [GENERAL.md](GENERAL.md).

**A different map, for dense input.** Writing the eigenproblem with the target
coordinate pinned makes it an algebraic *Riccati* equation whose step expansion
is exactly quadratic — and IPT turns out to be the fixed point that discards
both the rank-one and the quadratic term. Restoring them costs **no extra
matvec** (the Jacobian is diagonal-plus-rank-one, so Sherman–Morrison inverts
it in O(n)), and the resulting map is self-consistent Brillouin–Wigner: the
denominators use the *updated* eigenvalue, so a level sitting on top of the
target is no longer a pole. `bw_eig_partial` solves **106 of 240** random
instances against `ipt_eig_partial`'s **70**, at the same iteration count
(`bench_riccati.py`). It is not a strict superset — 2 of 240 go the other way —
and it is the wrong tool on sparse input, where the matvec is too cheap to
amortize its O(nk) inner loop. See [GENERAL.md](GENERAL.md).

`window_eig(A, lo, hi)` computes ALL eigenpairs in an interval via
purification (matmul only, no factorization, no target guess) with a
*certified* count — unlike ARPACK shift-invert, which needs `k` guessed up
front and silently returns fewer than the true count if you guess low
(measured: guessing half the true count returns exactly that many, with no
warning that any were missed). Not fast (6–7× slower than ARPACK even when
handed the true count, and `precision="mixed"` only helps past N~800) — use
it when the count must be right, not merely fast.

For the hard remainder — dense, non-normal, far from diagonal — `sdc_eigvals`
(spectral divide and conquer via the matrix sign function) solves it to 1e-13
with no basin condition, though at 0.13–0.22× of `dgeev` on this CPU. GENERAL.md
gives the exact break-even condition.

Both wins are *dispatchable*: applicability is decided by
$\rho = \max|W_{ij}|/|d_i-d_j|$, which `ipt_rate` computes in $O(N^2)$ —
free next to a single gemm — so a caller can pick IPT or LAPACK correctly
every time. [GENERAL.md](GENERAL.md) also records what does **not** work for
nonsymmetric matrices, and why.

## Usage

```python
import numpy as np
from ssj import (ssj_eigh, ipt_eigh, ipt_eig, ipt_eig_partial,
                 ssj_ipt_eigh, normal_eig, refine_eig)

# near-diagonal symmetric input: a handful of gemms, beating dsyevd
w, V = ipt_eigh(A_near_diagonal)

# GENERAL (nonsymmetric) near-diagonal input: 4-12x faster than dgeev
w, V = ipt_eig(A_general)          # eigenvectors are NOT orthogonal

# normal but nonsymmetric (A A^T = A^T A): exact, via ONE Hermitian solve
w, U = normal_eig(A_normal)        # U unitary, w complex

# k targeted eigenpairs (interior is as cheap as extremal), O(N^2 k)
w, V = ipt_eig_partial(A, cols=[500, 501, 502])

# is IPT applicable? O(N^2), free next to a gemm
from ssj.ipt import ipt_rate
use_ipt = ipt_rate(A) < 0.5

# any symmetric input: SSJ globalizes, IPT finishes cheaply
w, V = ssj_ipt_eigh(A)

A = np.random.randn(500, 500); A = (A + A.T) / 2

w, V = ssj_eigh(A)                      # eigenvalues ascending, eigenvectors in columns
w, V = ssj_eigh(A, method="gemm")       # factorization-free: every flop a gemm
w, V, info = ssj_eigh(A, return_info=True)   # info["sweeps"], info["history"], ...
```

Complex Hermitian matrices work with the same call (anti-Hermitian generator,
unitary retraction).

**GPU:** pass a CuPy array and the whole iteration runs on the device — the
implementation is backend-agnostic between NumPy and CuPy. `method="gemm"` is
the natural GPU choice (matmuls and elementwise maps only). `bench_gpu.py`
measures SSJ against cuSOLVER's `syevd` on your GPU; see BENCHMARKS.md for
what to expect and why the CPU verdict may reverse there.

### Methods and accelerations

| `method` | retraction | when to use |
|---|---|---|
| `"auto"` (default) | Householder QR, switching to adaptive-depth Newton–Schulz once ‖K‖_F < ½ | general use |
| `"qr"` | Householder QR every sweep | reference / strictest orthogonality path |
| `"gemm"` | spectral cap ‖K‖₂ ≤ 1 + adaptive Newton–Schulz | hardware where gemm far outruns factorizations; ~2× flops of `"auto"` at the same sweep count, all of them gemms |
| `"cholqr2"` | CholeskyQR2 (gemm + small triangular ops) | stack-dependent; measured *slower* than QR on our CPU BLAS — see BENCHMARKS.md |

- `precision="mixed"` runs the linear phase in float32 and hands off to
  float64 for the quadratic tail — full final accuracy, ~1.3–1.4× on CPU,
  much more on tensor-core GPUs. Safe because the map is memoryless.
- `prologue=k` runs k unshifted QR-algorithm steps before the first sweep —
  collapses the sweep count for graded/decaying spectra (measured 45 → 5
  with `prologue=3`), does nothing for flat spectra.
- `X0=` warm-starts from a nearby eigenbasis (tracking: 1–5 sweeps).

Measured dead ends (see BENCHMARKS.md): over-relaxation γ·K slows or
diverges for every γ > 1, and generator-space momentum slows for every β
tried — the map tolerates no memory and no extra aggressiveness.

## Repository layout

- `src/ssj/core.py` — the solver (~200 lines of NumPy)
- `validate.py` — reproduces the convergence battery and scaling table from RESULTS.md
- `experiments.py` — mechanism experiments: trajectory, monotonicity, and controlled divergence of the variants that remove a saturation
- `tests/test_ssj.py` — correctness tests (run with `pytest` or directly)
- `src/ssj/ipt.py` — IPT and the SSJ->IPT hybrid (symmetric and general)
- `src/ssj/normal.py` — normal-matrix solver and the norm-reducing shear
- `src/ssj/sdc.py` — spectral divide and conquer (general, globally convergent)
- `experiments_shear.py` — the shear/normality explorations
- `bench_vs_lapack.py` — head-to-head against LAPACK
- `bench_sparse.py` — the large-sparse win, and `--envelope` for its boundaries
- `bench_screen.py` — how well the one-hop screen predicts convergence (badly)
- `bench_riccati.py` — the Brillouin-Wigner map head-to-head against IPT
- `experiments_general.py` — the nonsymmetric explorations behind GENERAL.md
- `ALGORITHM.md` — algorithm specification + implementation notes
- `GENERAL.md` — the general (nonsymmetric) problem: one win, three failures
- `RESULTS.md` — original measured results (Julia reference implementation)
- `BENCHMARKS.md` — measurements of this implementation

## Requirements

NumPy. SciPy is optional (used only for the LAPACK `trtri` inside the
`"cholqr2"` option). Python ≥ 3.9.

## Honest context

**SSJ** is not CPU-competitive with LAPACK from a cold start on dense random
input (`dsyevd` wins by ~30× at N=1000 on our box; see BENCHMARKS.md). Its
niches are hardware where an eigensolve costs many gemm-equivalents and the
`"gemm"` variant's pure-multiplication diet applies, and robustness — a
three-line, parameter-free method with an empirically global basin and native
degeneracy handling. No convergence proof is known; see the "Not established"
section of RESULTS.md.

**IPT** is the part that beats LAPACK, but only inside its basin
(ρ ≲ 0.1), which requires well-separated eigenvalues and not merely small
coupling — see GENERAL.md. Outside the basin it diverges, and reports that
rather than returning a wrong answer. Tracking a perturbed matrix and dense
random input both *lose*, measured, in BENCHMARKS.md.

All timings come from one shared 4-core container with ~30% run-to-run
noise; gemm-equivalent counts are the noise-free comparison and are reported
alongside. GPU numbers are predictions, not measurements.
