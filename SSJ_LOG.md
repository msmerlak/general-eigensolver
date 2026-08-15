# SSJ improvement log

A dedicated track, separate from `MAP_LEDGER.md`. That ledger hunts for *new
maps*; this one improves the map the repository already ships. **Append here
before anything else**, one line per attempt, negative results included.

## What SSJ is, and where the cost sits

```
X ← I
repeat
    B ← XᵀA X ;  d ← diag(B)
    K_ij ← ½·atan( 2B_ij / (d_j − d_i) )      (±π/4 at zero gap; antisymmetric)
    X ← orth( X(I + K) )
```

Per sweep: 2 gemms (form B) + 1 orthogonalization (QR ≈ 0.67 gemm-equivalents,
or an adaptive Newton–Schulz endgame) ≈ **2.67 gemm-equivalents**.

Baseline measured for this log (GOE, sequential, min-of-3, `np.linalg.eigh` as
the LAPACK reference):

| n | sweeps | gemm-equiv | residual | wall vs `eigh` |
|---|---|---|---|---|
| 200 | 20 | 53.4 | 1.3e-15 | 13.5× |
| 400 | 24 | 64.1 | 1.2e-15 | 30.9× |
| 800 | 29 | 77.4 | 1.3e-15 | 18.1× |

Wall ratios on this container are noisy (~30%; the n=200 row is a clear
outlier) — **use sweeps and gemm-equivalents as the hardware-free measure**.

Two facts the docs understate and that matter for choosing a direction:

* **The sweep count GROWS with n** — 20 → 24 → 29 over a 4× size range, not the
  "13–20" quoted elsewhere. So SSJ is not merely a constant factor off
  `dsyevd`; the constant itself is drifting upward. Anything that flattens that
  growth is worth more than a fixed-factor saving.
* LAPACK `dsyevd` costs **8.3–18 gemm-equivalents** at N=1000, so even at n=200
  SSJ is ~3–6× off in the hardware-free unit before implementation quality is
  considered. The ~30× wall gap is that, times BLAS tuning.

## The mechanism, established by measurement this session

Do not propose changes that break these — each was probed independently and
found load-bearing:

* **The divide-by-gap generator is where the speed lives.** Replacing it with a
  gradient (Brockett double bracket) costs ~800× even after momentum
  acceleration; the QR and Cholesky-LR flows cost 7–10×.
* **The arctan saturation is not a safety decoration, it is the price of the
  fast generator.** Continuation (homotopy) keeps denominators small but cannot
  make them nonzero, and dies on exact degeneracy where SSJ's saturation
  survives it natively.
* **The symmetric pairing gives the descent property.** A one-sided variant
  aimed at the Schur form loses it and stalls on ~1 instance in 12.
* **ρ(J) is a diagonal-similarity invariant**, so no coordinate reconditioning
  can change any locator's basin.

## Known accelerations (already shipped)

| lever | effect |
|---|---|
| `prologue=k` (unshifted QR steps first) | graded/decaying spectra 45 → 5 sweeps; nothing on flat spectra |
| `precision="mixed"` | ~1.3–1.4× on CPU; more on tensor-core GPUs |
| `X0=` warm start | tracking a perturbed matrix: 1–5 sweeps |
| `method="gemm"` | factorization-free; ~2× flops at equal sweeps, all gemm |
| adaptive Newton–Schulz endgame | replaces QR once ‖K‖_F < ½ |

## Measured dead ends — do not retry

| attempt | result |
|---|---|
| over-relaxation γ·K, γ > 1 | slows or diverges for **every** γ tried |
| generator-space momentum | slows for every β tried (the saturation is what it breaks) |
| Anderson acceleration | diverges (RESULTS.md, independently reproduced) |
| second-order retraction | no gain |
| deferred orthonormalization | no gain |
| CholeskyQR2 retraction | slower than QR on this CPU BLAS |

## Open targets, roughly by expected value

1. **Cut the sweep count** (13–20 today). This is the dominant term and the
   most valuable direction. Shifts, deflation of converged columns, a better
   first sweep, anything that reduces iterations without breaking the mechanism.
2. **Cheapen the retraction** (0.67 of the 2.67 per sweep). The Newton–Schulz
   endgame already does this late; doing it earlier or cheaper is open.
3. **Exploit structure** — banded/sparse input, where forming XᵀAX densely is
   wasteful.
4. **Deepen mixed precision** beyond the current 1.3–1.4×.
5. **A convergence proof.** None is known. Genuinely valuable and the one item
   here that is a theory contribution rather than an engineering one.

## The GPU question, and how to answer it

`ssj_gpu_colab.ipynb` (repo root) is a standalone Colab notebook that measures
the one thing this repo cannot measure on its own hardware: **whether the
SSJ-to-incumbent ratio shrinks on a GPU.** The argument for expecting it to is
structural — with `method="gemm"` every SSJ flop is a matmul, while `syevd`
spends much of its time in memory-bound `symv` inside the tridiagonal
reduction, which GPUs execute badly.

It reports a **difference of ratios** (SSJ/LAPACK on CPU vs SSJ/cuSOLVER on
GPU), measured in one session, so the answer cannot be confounded by which
machine Colab hands out. It also prints cuSOLVER's cost in gemm-equivalents,
which alone decides the outcome: SSJ needs 55–80, so unless cuSOLVER rises
above that on the incumbent side, SSJ cannot win the cold solve.

Three things worth knowing before reading its output:

* **Consumer GPUs are fp64-crippled** (T4/P100 run fp64 at 1/32 of fp32;
  A100/V100 at 1/2). Cell 1 *measures* the ratio rather than assuming it and
  says which rows are meaningful. A null result on a T4 is not a null result.
* **The cold solve is SSJ's worst case.** The notebook also measures
  warm-started tracking, where SSJ takes 1–5 sweeps and `syevd` has no way to
  accept a warm start at all. That is the only regime where a ratio below 1 is
  plausible, and it is the claim most likely to survive.
* The notebook carries a **standalone copy** of the solver (the repo is
  private, so Colab cannot clone it). Two deliberate GPU-specific deviations,
  both marked in its source: the power iterations drop their per-iteration host
  syncs (~125 per sweep), and the block pass is batched.

**Validated before shipping, on the NumPy path** (the code is
backend-agnostic, so NumPy exercises what CuPy would): all 30
config × spectrum combinations converge with residual ≤1e-14 and
orthogonality ≤1e-12; the batched block pass reproduces the CPU SSJ-BC sweep
counts *exactly* (exact 5-fold degeneracy 69→25, clustered-1e-9 33→9, GOE
n=400 24→10); and the block pass is strictly monotone in off(B) — measured
increase exactly 0.0 — including at n=300, where m=32 does not divide n and
the tail-exclusion path fires.

## Attempts

| # | attempt | verdict |
|---|---|---|
| 1 | **SSJ-BC**: block-cluster preconditioner (block-Jacobi pass on sorted-diagonal blocks) + mass-capped angle gate + Newton–Schulz target floor | **real, verified independently — the first genuine improvement.** See below. |

### 1. SSJ-BC — verified

The agent first instrumented *why* the sweep count grows: `diag(B)` must climb
from spread ‖A‖/√n to the true spectral spread, costing ≈½·log n sweeps, and
while the spread is small nearly every gap is comparable to every coupling —
7697 pairs saturate at ±π/4 at n=400 and the contraction rate sits at 0.99.
The block pass hands the iteration ~√m of that spread for free.

**Reproduced independently against the shipped `ssj_eigh`** (sweeps are
load-independent; wall measured with the box quiet):

| case | shipped | BC m=32 | BC m=n/8 |
|---|---|---|---|
| GOE n=200 | 20 | **9** | 9 |
| GOE n=400 | 24 | **11** | 10 |
| GOE n=800 | 29 | **14** | 10 |
| exact 5-fold degeneracy n=200 | 69 | 25 | 26 |
| clustered 1e-9 n=200 | 33 | **9** | 9 |

Wall, quiet machine, min-of-3: **1.21× / 1.38× / 1.46×** at m=32 and
**1.27× / 1.44× / 1.75×** at m ∝ n, for n = 200/400/800 — the margin *grows*
with n. Residuals 1.2e-15…1.9e-14, orthogonality ≤2.8e-14 throughout.

**The growth flattening is the real result.** Shipped sweeps go 20 → 24 → 29
over n=200→800; with m ∝ n they go **9 → 10 → 10**. That is the term the
baseline section flags as mattering most.

**Two corrections to the proposing agent's report**, both found by re-measuring:

* **The degeneracy claim does not reproduce at its stated size.** It reported
  70 → 8 sweeps (8.75×) on exact 5-fold degeneracy; on my construction I get
  69 → 25 (2.8×). Still a real gain, and still the largest of any spectrum
  tested — but a third of the claim. Constructions differ, which is exactly why
  the number needs restating rather than inheriting.
* **The shipped defaults are the configuration the agent discarded.**
  `ssj_bc` defaults to `block_m=16, block_passes=1, part="gap"`, but every
  measured result used `m=32, passes=2` with *equal-chunk* partitioning — and
  the report explicitly finds gap-partitioning **worse**. With the defaults the
  wall gain vanishes entirely (1.02×, 1.00×), which is how I first mismeasured
  this. Any integration must ship the tested configuration as the default.

**Not yet integrated**, deliberately. It is not a strict win — a tight `X0`
warm start regresses 2 → 3 sweeps, because the block pass fires on an iterate
that is already nearly converged. So it belongs behind an option, default off,
until that case is handled. It also required a companion fix (Newton–Schulz
target floored at `0.1·tol`); without it the faster convergence outruns a
tolerance calibrated for the old rate and silently loses 2–3 digits — a good
reminder that an acceleration can break a downstream tolerance.

**What it is, stated honestly:** a *preconditioner around* SSJ, not a change to
the map. The generator, the arctan saturation and the symmetric pairing are
imported unmodified. With m ∝ n it is "SSJ preconditioned by one level of
block-Jacobi", which the agent says plainly and which should not be described
as "SSJ got faster". And SSJ remains 24–29× LAPACK even with it.
