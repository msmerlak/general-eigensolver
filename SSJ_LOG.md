# SSJ improvement log

A dedicated track, separate from `MAP_LEDGER.md`. That ledger hunts for *new
maps*; this one improves the map the repository already ships. **Append here
before anything else**, one line per attempt, negative results included.

## What this is teaching us about the eigenvalue problem

*(rewritten each tick; as of attempt #14)*

**Fast eigensolvers divide by gaps**; the price is a basin (ρ < 1) or a
saturation. **Two phases**: spread-limited globalization (injectable — the
schedule, #9; head blocks gated off once rel_off < 0.3, #14) and a
manifold-free endgame past the cliff (IPT, #8). **The merge is irreducibly
global** (#10). **Gains compose by Amdahl** (#11). **The algorithm stands at
~2–3× LAPACK in flop units; the rest of the CPU wall gap is substrate**
(#13).

**The tracking niche died on CPU and moved to GPU (#14).** The claim "warm
starts beat re-solving" was substrate-blind. Measured on a clean box across
ε = 1e-8…1e-1 at n=800: warm never beats LAPACK — even one sweep from a
perfect basis costs 2.6× a full `dsyevd` re-solve, because one SSJ sweep
(~14.5 measured ge) already exceeds LAPACK's whole solve (~13.8 ge here).
No warm start can win where a sweep costs more than the incumbent's full
solve. The niche exists exactly where that inequality flips: substrates
whose incumbent is expensive relative to a gemm (cuSOLVER at 30–40 ge).
Every "where SSJ wins" claim is a claim about a substrate, not the
algorithm.

**Two warm-path fixes shipped anyway** (#14, unconditionally right): entry
QR is now gated by a 1-ge Gram check (it re-orthonormalized an
already-orthonormal previous eigenbasis at 12 ge — the largest single item
in a warm solve), and the schedule's head blocks are gated on rel_off < 0.3
(state-based, not X0-based: a bad X0 still fires them; a warm start or late
sweep never pays an n/2 eigh for spread it has).

**Method lessons now four of a kind:** untested defaults (#2), unasserted
outputs (#5/#7), unexamined library identity (#13), and **cross-era number
reuse** (#14: "expected 8 sweeps" came from the chained prototype, not the
in-solver path — comparing across code eras manufactured a phantom
regression).

**Open:** (1) the GPU run — now carrying schedule+hybrid+mixed AND the two
warm fixes belong in the notebook next sync; tracking is the notebook's
strongest card and now provably GPU-only; (2) era-stale: deferred
orthonormalization at 8-sweep economics; (3) a convergence proof.

## What SSJ is, and where the cost sits

```
X ← I
repeat
    B ← XᵀA X ;  d ← diag(B)
    K_ij ← ½·atan( 2B_ij / (d_j − d_i) )      (±π/4 at zero gap; antisymmetric)
    X ← orth( X(I + K) )
```

Per sweep, *by flop count*: 2 gemms (form B) + 1 orthogonalization (QR ≈ 0.67
gemm-equivalents, or an adaptive Newton–Schulz endgame) ≈ 2.67
gemm-equivalents.

> **That model is wrong by 6.6×, and much of this log was calibrated on it.**
> Attempt #5 profiled a sweep: at n=800 it is **17.7 gemm-equivalents**, not
> 2.67. `_orth_qr` alone costs 8.73 (LAPACK QR runs far below gemm efficiency)
> and `_angles` costs 4.82 (a dozen n² elementwise passes, one of them a
> transcendental). Only `form B` (2.04) and `Y = X(I+K)` (1.29) match the
> model. Use the flop model for *asymptotics*; use measured gemm-equivalents
> for anything that decides a design.

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
* **The polar retraction's saturation is load-bearing too**, not just the
  arctan on the angles. Swapping the polar factor for Cayley — which rotates
  each K-plane by 2·arctan(σ/2) instead of arctan(σ), saturating at π rather
  than π/2 — costs 20→53 sweeps at n=200, 24→266 at n=400, and fails to
  converge at all at n=800. Both saturations are needed, and the weaker one
  degrades *with n*.
* **The symmetric pairing gives the descent property.** A one-sided variant
  aimed at the Schur form loses it and stalls on ~1 instance in 12.
* **ρ(J) is a diagonal-similarity invariant**, so no coordinate reconditioning
  can change any locator's basin.

## Known accelerations (already shipped)

| lever | effect |
|---|---|
| `prologue=k` (unshifted QR steps first) | graded/decaying spectra 45 → 5 sweeps; nothing on flat spectra |
| `precision="mixed"` | ~1.3–1.4× on CPU; more on tensor-core GPUs |
| `X0=` warm start | 1–5 sweeps on a perturbed matrix — but on CPU this never beats a LAPACK re-solve (#14); the tracking case is GPU-only |
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
| CholeskyQR2 retraction | slower than QR on this CPU BLAS (18.4 gemm-equiv vs 9.2 at n=800) |
| Cayley retraction (I−K/2)⁻¹(I+K/2) | **does not converge** — n=800 hits the 1000-sweep cap at Δλ 4.8e-2 |

## Open targets, roughly by expected value

1. **Cut the sweep count** (13–20 today). This is the dominant term and the
   most valuable direction. Shifts, deflation of converged columns, a better
   first sweep, anything that reduces iterations without breaking the mechanism.
2. **Cheapen the retraction.** Now the single largest term by a wide margin:
   measured at **8.73 of a sweep's 17.70 gemm-equivalents (49%)** at n=800, not
   the 0.67 the flop model claimed. **No cheaper retraction has been found.**
   Newton–Schulz appeared to cost 4.29 but that measurement was invalid (see
   attempt #7); verified, it costs 19–27 — *more* than QR — on the early
   sweeps where ‖K‖₂ is large. Cayley does not converge. Anything that makes
   the retraction genuinely cheaper remains the highest-value target here, and
   nothing tried so far does.
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
| 2 | **Integrate SSJ-BC into `src/ssj/core.py`** behind `block_m`, default off; batched block pass; test the `block_until` gate | **shipped.** Sweep gains reproduce exactly. Two inherited claims did *not* survive re-measurement — see below. |
| 3 | **Wall time for SSJ-BC**, the measurement owed by #2 — and the dense-`Qfull` defect it exposed | **fixed and measured.** 1.27–2.11× wall. The block application was costing `n/m`× the flops it needed to. |
| 4 | **Attack the block pass's memory-bound overhead** — config grid, then a component-level breakdown | **mostly negative.** The hoped-for 1.35× is not there. Config tuning is a flat optimum; one real fix found (n=800 1.38× → **1.47×**). |
| 5 | **Profile the sweep body**; test whether the cheaper retraction wins | **cost model corrected (6.6× off); one hypothesis refuted, one real 1.60× win.** `method="gemm"` + BC is now the fastest configuration. |
| 6 | **Rebuild `_angles` in place**; O(n log n) tie detection instead of O(n²) | **shipped, bit-identical.** 2.0–2.9× on the map itself, **1.04–1.09× end to end** — diluted by Amdahl. |
| 7 | **Cheapen the retraction** (target #2, 49% of a sweep): Cayley transform; re-verify the Newton–Schulz comparison | **negative, and it invalidates a number in #5.** Cayley does not converge at n=800. QR is the cheapest verified retraction. |
| 8 | **Compose the globalizer with the manifold-free endgame** (`ssj_ipt_eigh` + BC), and instrument where the basin opens | **shipped, with a real bug found by the output assert.** The basin opens as a *cliff*; hybrid+BC = 12 sweeps + 6 gemms at n=800. |
| 9 | **Anatomy of the pre-cliff phase, then a block-size SCHEDULE** (big blocks early, small late) | **the campaign's best result.** GOE n=800: 15 → 8 sweeps; wall 1.90× over plain, 2.13× with the hybrid — 13.5–15.9× LAPACK, from 28–34×. |
| 10 | **Is the merge local?** K/B mass vs sorted distance; block-passes-only solver; banded-K map | **decisively no — with the campaign's sharpest mechanism finding.** Angles are local, couplings are not; rotation mass ≠ annihilation work. |
| 11 | **The full composition: mixed × schedule × hybrid**, plus the fp64-phase block policy it needed | **champion config, 2.33× over plain (16–18× LAPACK) — and gains compose by Amdahl, not multiplication.** |
| 12 | **Bring the GPU notebook up to the composed solver** (schedule, predictive NS, IPT hybrid, mixed segments) | **shipped and validated** — 50 config × spectrum rows pass on the NumPy path; hybrid runs GOE n=200 in 4 sweeps + 8 gemms. |
| 13 | **Audit the floor claim**: decompose measured-vs-modelled per component; test symmetric BLAS (syrk/syr2k, BLAS-level CholeskyQR2) | **floor claim corrected — algorithm is ~2–3× LAPACK in flop units; substrate ~6×, not reclaimable from Python on this stack.** scipy/numpy BLAS mismatch caught by control. |
| 14 | **Tracking**: warm-start crossover chart; schedule-head and entry-QR defects | **the CPU tracking niche is dead — warm never beats LAPACK here** (2.6× at best). Two warm-path fixes shipped; niche relocated to GPU, where the notebook tests it. |

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

### 2. SSJ-BC integrated — and two inherited claims that did not survive

Shipped in `ssj_eigh` as `block_m` (0 = off, unchanged default), with
`block_passes=2` and `block_until=0.0`. `test_default_is_unchanged` asserts
`block_m=0` reproduces the untouched iteration bit for bit. 131 tests pass
(19 new, in `tests/test_block_cluster.py`).

**The block pass was rewritten, not ported.** The prototype loops over blocks
in Python, issuing `n/m` small eigensolves and `2n/m` small gemms — latency-
bound, and badly so on GPU. The shipped version rolls the sorted diagonal
cyclically by `offset` instead of cutting a ragged head and tail, so every
block has size exactly `m`, **one batched `eigh`** handles all of them, and
the block-diagonal factor is assembled once so applying it is two gemms.
Because it is a rewrite, none of attempt #1's numbers were inheritable; all
were re-measured.

Sweep counts (`method="auto"`, GOE seed 1), against attempt #1's independently
verified column:

| case | shipped default | integrated BC m=32 | attempt #1 verified |
|---|---|---|---|
| GOE n=200 | 20 | **9** | 9 |
| GOE n=400 | 24 | **11** | 11 |
| GOE n=800 | 29 | **14** | 14 |
| exact 5-fold deg n=200 | 69 | **25** | 25 |
| exact 5-fold deg n=500 | 74 | **31** | — |
| clustered 1e-9 n=200 | 33 | **9** | 9 |
| zero diagonal n=200 | 21 | **9** | — |

An exact match on every case attempt #1 covered. Accuracy holds throughout:
|Δλ|/‖A‖ ≤ 6e-15, residual ≤ 4.5e-14, orthogonality ≤ 2.9e-14. Also verified
on `gemm` (25 → 10), `cholqr2`, `mixed` precision, and complex Hermitian.

**Claim that did not reproduce — the warm-start regression.** Attempt #1
records "a tight `X0` warm start regresses 2 → 3 sweeps", and that single fact
is why it was left unintegrated. It does not happen here. Measured, with
`X0` = the exact eigenbasis of a nearby matrix:

| perturbation | default | BC gated | BC ungated |
|---|---|---|---|
| ε=1e-6, n=200 | 2 | 2 | 2 |
| ε=1e-4, n=200 | 3 | **2** | **2** |
| ε=1e-6, n=400 | 2 | 2 | 2 |
| ε=1e-2, n=400 | 5 | **3** | **3** |

BC does not regress the warm start — it *improves* it, and gating changes
nothing. Whether the difference is the batched partitioning or the prototype's
`part="gap"` default is not established; what is established is that the
blocker is absent from the shipped implementation.

**Consequence: `block_until` defaults to 0.0, not the 1e-3 I first wrote.**
I added the gate to protect against the regression above, then measured what
it costs. It is a pure loss:

| case | gate 0.0 | gate 1e-6 | gate 1e-3 |
|---|---|---|---|
| exact 5-fold deg n=200 | **25** | 25 | 38 |
| exact 5-fold deg n=500 | **31** | 38 | 51 |
| clustered 1e-9 n=200 | **9** | 9 | 12 |
| GOE n=400 | **11** | 11 | 12 |

The gate switches the preconditioner off precisely on the spectra it exists
for — tight clusters are still being resolved well below rel_off 1e-3. Kept as
a knob, defaulted off. **Had I shipped the default I reasoned my way to rather
than the one I measured, degenerate spectra would have lost a third of the
gain silently** — the same shape of error as attempt #1's untested defaults.

**Wall time: measured in attempt #3**, which also found and fixed a defect in
what this attempt shipped. The sweep ratio is *not* a wall ratio — a BC sweep
buys a batched `eigh` and the cost of applying it.

### 3. Wall time — and the defect measuring it exposed

Attempt #2 shipped sweep counts and owed a wall figure. Measuring it found
that **most of the sweep gain was being handed straight back.**

**The defect.** #2 assembled the block-diagonal factor into a dense
`(keep, keep)` matrix so that applying it was "two gemms rather than 2·nb
small ones". That is a flop *pessimization*: a dense gemm against a
block-diagonal operator does `n/m` times the necessary work. Modelled, in
n³-gemm units, per pass:

| n | batched eigh | dense `Qfull` | block-diagonal | waste |
|---|---|---|---|---|
| 200 | 0.319 | 2.765 | 0.461 | 6.0× |
| 400 | 0.080 | 2.765 | 0.230 | 12.0× |
| 800 | 0.021 | 3.000 | 0.120 | 25.0× |

A sweep is ~2.67 gemm-equivalents and `block_passes=2`, so the dense form was
adding ~6.0 — **more than doubling the cost of every sweep**. This is
load-immune arithmetic and needs no quiet box to trust.

**The fix.** Apply Q batched over the `(nb, m, m)` stack via reshape and
transpose, with no Python loop and no dense factor. Wall, before → after:

| case | before fix | after fix |
|---|---|---|
| GOE n=200 | 1.10× | **1.27×** |
| GOE n=400 | 1.12× | **1.39×** |
| GOE n=800 | 1.15× | **1.38×** |
| 5-fold deg n=200 | 1.46× | **1.67×** |
| clustered 1e-9 n=200 | 1.91× | **2.11×** |

131 tests still pass, including block-pass monotonicity and orthogonal
similarity, so the batched application is exactly equivalent.

**Measurement method**, since the box is shared and was not reliably quiet:
the script blocks until the 1-minute load average is under 0.40, interleaves
default / BC / LAPACK *within each repetition* so drift lands on all three
equally, takes min-of-7, and re-times LAPACK at the end as a contamination
check. That check reported 16–17% drift — **downward** (LAPACK 21.6 → 18.0 ms
at n=400), i.e. the box got quieter as the run progressed, so the absolute
`BC/LAPACK` column carries ±15% but the interleaved speedup column does not.
For calibration, LAPACK here runs 1.15–1.25× the quiet-box reference recorded
at the top of this log.

**Still unexplained, and the next target.** At n=800 the flop model predicts
1.87× (77.4 gemm-equivalents against 41.3) but wall delivers 1.38×. The gap is
that a block pass is **memory-bound, not flop-bound**: it does two full n²
gathers (`B[p][:, p]`) plus the reshape/transpose copies, none of which the
flop model counts. Removing or fusing those gathers is the concrete next
improvement, and would be worth ~1.35× on top of what is measured here.

**Where this leaves SSJ-BC:** 17–31× LAPACK on GOE, against 20–46× for the
shipped default. A real improvement, and still not competitive on a cold dense
solve — which is what the GPU notebook exists to re-ask on hardware where the
incumbent is weaker.

### 4. The memory-bound overhead — mostly a negative result

Attempt #3 logged the gap between the flop model (1.87× at n=800) and measured
wall (1.38×) as memory traffic, and estimated ~1.35× available from removing
it. **That estimate was wrong.** Two lines of attack, one small win.

**Config tuning is a flat optimum — do not retry.** A block pass costs
arithmetic that grows with `m` while the sweep saving also grows with `m`, and
the two cancel almost exactly. Modelled total gemm-equivalents over the grid:

| config | GOE n=400 | GOE n=800 | 5-fold deg n=200 | clustered n=200 |
|---|---|---|---|---|
| default | 64.1 | 77.4 | 184.2 | 88.1 |
| m=32 p=1 | 41.7 | 47.8 | **93.2** | 41.4 |
| m=32 p=2 | **36.2** | 41.3 | 105.8 | **38.1** |
| m=64 p=1 | 41.4 | 44.7 | 126.6 | 43.8 |
| m=64 p=2 | 38.1 | 39.5 | 176.7 | 49.5 |
| m=128 p=2 | 49.5 | **38.1** | 267.7 | 60.9 |

The best config differs per case and the optimum is shallow (36–41 across most
of the grid at n≥400). The hypothesis that `(m=64, p=1)` would beat
`(m=32, p=2)` by halving the permutations is **false** — it is worse at three
of four cases. `m=32, passes=2` is a fine default; there is no tuning win here.

**The flop model understates a block pass by ~21×.** Measured components, one
pass at n=800, m=32, in units of one n³ gemm (quiet box, min-of-7):

| component | cost | note |
|---|---|---|
| permute B (two-sided) | 0.68 | pure memory traffic |
| permute X | 0.43 | pure memory traffic |
| extract blocks | 0.00 | free |
| batched `eigh` | 0.71 | ~130 µs per 32×32 block — call overhead, not flops |
| apply ×3 | 1.14 | |
| **total** | **2.96** | **model said 0.141** |

So the overhead is not one removable copy; it is spread across permutation,
LAPACK call overhead, and the applies. Two levers tested against it:

* **Batched `eigh` vs a Python loop of `eigh` calls: a wash** (within noise at
  every size). The ~130 µs per 32×32 block is LAPACK entry cost and is not
  recoverable by restructuring.
* **The apply: a real but small win.** Attempt #3 chose the batched
  reshape/transpose form on the assumption that a Python loop is bad. On
  *NumPy* that assumption is wrong — each column block is a contiguous slice,
  so a loop of `nb` gemms writes the target in place with no temporary at all,
  and `nb` is only ~25. Measured 1.15–1.69× on the apply alone.

**Shipped:** the apply is now looped on NumPy and batched on CuPy, chosen by
backend, since a GPU genuinely does prefer the batched form (`nb` kernel
launches are latency-bound there). A test pins the two implementations against
each other on NumPy so the GPU branch cannot drift untested.

**Wall, quiet box, interleaved min-of-7** (contamination check 4.9–5.8% drift,
the cleanest run in this log; LAPACK calibrates at 1.07–1.13× the reference):

| case | attempt #3 | now |
|---|---|---|
| GOE n=200 | 1.27× | 1.27× |
| GOE n=400 | 1.39× | **1.41×** |
| GOE n=800 | 1.38× | **1.47×** |
| 5-fold deg n=200 | 1.67× | 1.65× |
| clustered 1e-9 n=200 | 2.11× | **2.19×** |

+6.5% at n=800, ~1–4% elsewhere, and nothing at n=200. A real gain, an order
of magnitude short of the 1.35× attempt #3 predicted. **The prediction was
wrong because it assumed the overhead was one identifiable copy; it is
distributed, and most of it is irreducible LAPACK and permutation cost.**

Sweep counts are unchanged except exact 5-fold degeneracy, which moved 25 → 26
— gemm blocking differs between the two applies, which shifts a borderline
convergence test by one sweep. Benign, and worth knowing when comparing runs.

**What is actually left at n=800:** the sweep body itself is ~110 ms while the
two block passes add ~55 ms. Halving the sweep count is therefore bought at
50% overhead per sweep, which is the whole story of the 1.47×. Further gains
have to come from the *sweep body*, not the block pass.

### 5. The sweep body — a corrected cost model, and a 1.60×

Attempt #4 concluded that further gains must come from the sweep body. Two
things came out of profiling it: the log's cost model is wrong, and the
Newton–Schulz floor added in #2 was quietly costing 1.6×.

**The cost model was understated 6.6×.** One sweep at n=800, measured against
a same-size gemm on a quiet box (min-of-7):

| component | measured | flop model |
|---|---|---|
| `_orth_qr` | **8.73** | 0.67 |
| `_angles` | **4.82** | ~0 |
| form B (2 gemms) | 2.04 | 2.0 ✓ |
| `Y = X(I+K)` (1 gemm) | 1.29 | 1.0 ✓ |
| symmetrize + off_frob | 0.68 | ~0 |
| **sweep total** | **17.70** | **2.67** |

The two gemm terms are accurate; everything else is not. QR runs far below
gemm efficiency, and `_angles` is a dozen n² elementwise passes. For reference
`_orth_cholqr2` measures 18.36 — on its own more than a whole sweep, which
independently confirms this log's existing dead-end entry for it.

**Refuted: "the cheaper retraction wins."** The profile shows
`_orth_ns_adaptive` at 4.29 against QR's 8.73, suggesting `method="gemm"`
(which uses Newton–Schulz every sweep, safe because its spectral cap keeps
σ(I+K) ≤ √2 < √3) should be ~2× faster. **End to end it is 1.05×.** The
profile timed NS at a *fixed* target of 1e-9; the real path uses an adaptive
target that tightens toward 1e-15, and adds a power iteration each sweep.
Generalizing from one unrepresentative operating point — the same error shape
as attempt #3's flop model.

**Found instead: the `_ns_target` floor fires from the first sweep.** With BC
active, #2 set the target to `min(base, 0.1·tol)` unconditionally. For `gemm`,
`base = 0.05·rel_off`, so at rel_off ≈ 1 the very first sweep was being asked
for 1e-14 orthogonality — several Newton–Schulz iterations nobody needs, every
sweep. The defect only has to be small on the *last* sweep, because the
product-form retraction re-measures and corrects it each time.

Now it tightens only when the contraction just observed predicts convergence
within `_NS_TIGHTEN_MARGIN · tol`. **Cold GOE, quiet box, interleaved
min-of-5** (contamination 7.4–8.8%):

| margin | n=800 gemm+BC | n=1200 gemm+BC | warm start ε=1e-6 |
|---|---|---|---|
| always-on (#2) | 2812 ms (1.00×) | 8061 ms (1.00×) | 2 sweeps |
| 1e12 | 1917 ms (1.47×) | 5419 ms (1.49×) | 2 sweeps |
| **1e9 (shipped)** | **1755 ms (1.60×)** | **4958 ms (1.63×)** | **2 sweeps** |
| 1e6 | 1805 ms (1.56×) | 5159 ms (1.56×) | 3 — **regression** |

1e9 is best on both axes. Too thin and a warm start beginning at rel_off ≈
3.6e-6 never tightens at all and loses a whole sweep; too wide and cold solves
tighten from sweep one and give the saving back. **Tightening a sweep early
costs a few NS iterations; tightening a sweep late costs an entire sweep** —
so the optimum sits wide.

**Accuracy is not paid for this.** The sensitive combination is a 1e-9 cluster
under `method="gemm"`, which without *any* tightening loses 2–3 digits
(|Δλ|/‖A‖ 4.5e-13, orthogonality 4.8e-12 — #2's claim, verified). At margin
1e9 it finishes at **2.5e-15 and 7.0e-15**, matching the always-on floor's
2.9e-15 and 9.5e-15. Across GOE, degenerate and clustered spectra at both
methods, |Δλ|/‖A‖ ≤ 6.6e-15 and orthogonality ≤ 1.5e-14.

Also verified, since #2 measured the floor for both paths: **the floor buys
`method="auto"` nothing** — identical sweeps and error with and without, on
all four spectra — because `auto` only reaches Newton–Schulz once ‖K‖_F < 0.5,
by which point its `rel_off²` target is already tight.

**`gemm` + BC is now the fastest configuration**, having been the slowest:
28.9–29.7× LAPACK against `auto` + BC's 30.1–36.5×, and 47× before this fix.
134 tests pass, 3 new — including one pinning `_ns_target` directly so the
early-loose/late-tight behaviour cannot silently regress.

### 6. `_angles` rebuilt in place — bit-identical, and Amdahl-limited

Attempt #5's profile made `_angles` the second-largest term in a sweep (4.82
gemm-equivalents at n=800, 9.29 at n=400 — a third of the sweep there). It had
never been optimized. Two structural observations, both testable:

* **Tie detection was O(n²) and needn't be.** Off the diagonal, `gap_ij = 0`
  exactly when `d_i == d_j`, so the entire question is answered by sorting the
  diagonal and checking adjacent pairs — **O(n log n)** — instead of building
  an n² mask with two comparisons, an AND and a reduction.
* **`nan_to_num` scanned all n² for nothing.** With no tied diagonal entries
  the only non-finite entries are ON the diagonal (`gap_ii = 0`), and
  `fill_diagonal` clears those regardless.

Plus the arithmetic is now built in place (`out=`, `/=`, `*=`) rather than
through four temporaries.

**`_angles` alone**, quiet box, min-of-9, in gemm-equivalents:

| n | shipped | rebuilt | speedup |
|---|---|---|---|
| 400 | 9.29 | **2.30** | 4.04× |
| 800 | 5.13 | **2.48** | 2.07× |
| 1200 | 3.49 | **1.23** | 2.85× |

On an exactly-degenerate spectrum — where the tie branch *does* fire, so only
the in-place arithmetic helps — it is still 2.0–3.9×. That was the surprise:
most of the win is the temporaries, not the skipped scans.

**Bit-identical output**, verified against a naive transcription of the formula
on ten cases: GOE, complex Hermitian, exact degeneracy, zero diagonal (every
pair tied), zero-diagonal complex, identity, all-zeros, pure diagonal, a single
zero coupling, and a single tied pair. Not "close" — `array_equal`. The
19 new tests in `tests/test_angles.py` assert that against the oracle, plus
exact anti-Hermiticity, the π/4 saturation bound, and that the input is not
mutated.

**End to end, however, 1.04–1.09×** (quiet box, interleaved min-of-5,
contamination 3.8% and 7.9%):

| config | n=800 | n=1200 |
|---|---|---|
| `auto` | 1.04× | 1.07× |
| `auto` + BC32 | 1.09× | 1.05× |
| `gemm` + BC32 | 1.06× | 1.08× |

(An n=400 run showed 1.10–1.19× but drifted 20.7% and is discarded.)

**This is Amdahl, and it was predictable from #5's profile**: `_angles` is 27%
of a sweep at n=800, so halving it caps the gain at ~1.16×. Worth having —
it is free, exact, and largest at the small-to-mid sizes where SSJ is most
competitive — but it does not move the standing. **The retraction is 49% of a
sweep and is where the remaining headroom is.**

One caution for reading this log: the `×LAPACK` column is not comparable
across attempts. LAPACK `eigh` at n=1200 measured 171.7 ms in attempt #5 and
214.1 ms here on the same box. Only the interleaved old/new ratios within a
single run are trustworthy.

### 7. The retraction — nothing cheaper exists, and #5 had a bad number

Target #2 is now the largest term in a sweep (49%), so this went after it.
Both results are negative, and one of them retracts a measurement from #5.

**Attempt #5's Newton–Schulz figure was invalid.** It reported
`_orth_ns_adaptive` at 4.29 gemm-equivalents against QR's 8.73, and this log
then re-ranked target #2 around "Newton–Schulz is the only cheaper
orthogonalization found". That measurement never checked its output. Re-run
with a verification step:

| retraction (n=800) | cost | ‖QᵀQ − I‖ |
|---|---|---|
| `_orth_qr` | 9.24 | 3.3e-14 ✓ |
| NS, target 1e-3 | 4.45 | **4.7e+08** ✗ |
| NS, target 1e-14 | 4.47 | **4.7e+08** ✗ |
| NS, capped + prescaled | **22.19** | 8.0e-15 ✓ |
| Cayley + `X @ Q` | 11.08 | 1.6e-13 ✓ |

Newton–Schulz **diverged**. On a first sweep ‖K‖₂ is 18–33, so
σ(I+K) = √(1+σ²) is far above the √3 convergence radius; the iteration blew
up, `_orth_ns_adaptive`'s stagnation guard (`dev > 0.9·prev`) tripped
immediately, and it returned fast. Fast because it failed. Done properly —
spectrally capped and prescaled, as `method="gemm"` actually does it — it
costs **19–27, i.e. 2–3× *more* than QR**, not less.

That also explains what #5 could not: why `method="gemm"` measured only 1.05×
end to end despite an apparently 2× cheaper retraction. There was no cheaper
retraction. (Why gemm is not *slower*: NS cost tracks ‖K‖₂, which is huge on
the first sweeps and small later, so a single-sweep number cannot settle it
either way. The 1.05× end-to-end figure stands as the trustworthy one.)

**Cayley: a clean dead end.** For anti-Hermitian K, `(I − K/2)⁻¹(I + K/2)` is
exactly orthogonal, replacing a QR with one linear solve — per-sweep, 1.14×
cheaper than QR at n=1200 (and *dearer* below). It is unusable anyway:

| case | `auto` | Cayley |
|---|---|---|
| GOE n=200 | 20 | 53 |
| GOE n=400 | 24 | **266** |
| GOE n=800 | 29 | **1000 — did not converge**, Δλ/‖A‖ 4.8e-2 |
| 5-fold deg n=200 | 69 | 73 |
| clustered 1e-9 n=200 | 33 | 67 |

The polar factor rotates each K-invariant plane by arctan(σ); Cayley by
2·arctan(σ/2), saturating at π instead of π/2. **That weaker saturation is
fatal, and gets worse with n** — the loss runs 2.7× at n=200, 11× at n=400,
divergent at n=800. Recorded above as a mechanism fact: this log already noted
the arctan on the *angles* as load-bearing; the saturation in the *retraction*
is a second, independent one. Reverted; no code shipped.

Factor-form drift showed up too, as this module's docstring predicts:
orthogonality ran 1.5e-13 → 3.4e-12 across n=200→800, against QR's ~1e-14,
because `X @ Q` never re-measures X's accumulated defect.

**Where target #2 now stands:** QR at 8.7–11.2 gemm-equivalents is the
cheapest *verified* retraction available. CholeskyQR2 (18.4), capped
Newton–Schulz (19–27) and Cayley (divergent) are all worse. The 49% remains
open, with three of the obvious candidates now closed off.

**Method note.** Two invalid measurements in this log have now had the same
cause: timing a numerical routine without checking it produced a right answer
(#5's Newton–Schulz here; #3's flop model was a different error). A timing
harness for anything iterative should assert on its output — a routine that
fails fast looks exactly like a routine that is fast.

### 8. The basin opens as a cliff — and the hybrid now composes with BC

First tick under the reflection mandate. The seed hint (IPT is pure gemm
because it has no manifold) points straight at the two-phase decomposition,
and the repo already shipped both halves — `ssj_ipt_eigh` (hand off to IPT
once ρ = max|W|/gap < 0.5) and SSJ-BC (a globalizer) — but they predate each
other: the hybrid's `ssj_kw` could not pass `block_m`, so the two
accelerations of this campaign had never met.

**Instrumented first: where does the basin actually open?** ρ per sweep on
the shipped code path:

| case | ρ<1 at sweep | total sweeps | post-gate share |
|---|---|---|---|
| GOE n=400 plain | 18 | 23 | 17% |
| GOE n=400 BC32 | 9 | 11 | 18% |
| GOE n=800 plain | 24 | 29 | 17% |
| GOE n=800 BC32 | 12 | 15 | 20% |

Two findings. **The basin opens as a cliff, not a ramp**: ρ sits above 99 for
most of the solve and collapses through 1 in a single sweep (BC at n=400:
18.1 → 0.04 in one sweep). So the gate threshold barely matters — 0.5 and 1.0
open on the same sweep — and there is no "partially open" regime to exploit.
**And BC does not enlarge the post-gate share** (~17–20% with or without): it
compresses both phases proportionally. The manifold-free endgame can only ever
replace that last ~20% of sweeps; the pre-cliff globalization is where 80% of
the cost lives, with the manifold genuinely load-bearing there (attempt #7).

**The composition shipped, and the output assert caught a real bug.** With
`block_m` threaded through, hybrid+BC first returned **1e-8 eigenvalue error**
on GOE — the coarse globalizing blocks run at `tol=1e-2`, the Newton–Schulz
floor keys off that coarse tol, and with BC the error falls through the gate
in one sweep, so the frame arrives at the hand-off orthonormal only to ~1e-7.
IPT inherits the frame verbatim and bakes the defect into the answer as a
similarity error. Without BC this never fired (K is still large at the gate,
so the last retraction was an exact QR) — a latent bug in the shipped hybrid,
exposed by the composition. **The lesson, stated as mechanism: leaving the
manifold is fine; leaving it uncleanly is not.** One exact QR at the hand-off
(once per solve) fixes it: hybrid+BC now lands at 4.3e-15 / 5.8e-15.

**End to end, load-immune** (accuracy asserted on every row):

| case | ssj | ssj+BC | hybrid | hybrid+BC |
|---|---|---|---|---|
| GOE n=400 | 24 sw | 11 sw | 20 sw + 18 it | **9 sw + 6 it** |
| GOE n=800 | 29 sw | 14 sw | 25 sw + 14 it | **12 sw + 6 it** |
| clustered 1e-9 n=200 | 33 sw | 9 sw | 32 sw + 8 it | **8 sw + 5 it** |
| 5-fold deg n=200 | 69 sw | 26 sw | 66 sw (fallback) | 23 sw (fallback) |

An IPT iteration is one gemm against a sweep's measured ~17.7 gemm-equivalents,
so trading 2 sweeps for 6 iterations plus one hand-off QR is a real saving.
Degeneracy correctly never opens the gate (ρ = ∞ on exact ties) and falls back
to pure SSJ+BC unharmed. Caveat kept honest: IPT eigenvectors are only
implicitly orthogonal — on the 1e-9 cluster the hybrid's ortho is 6.5e-12
(vs SSJ's 4.2e-15), with residual and eigenvalues at full precision.

**Wall: inconclusive this tick.** Both runs contaminated (14–24% drift; the
box would not stay quiet). Interleaved ratios *suggest* hybrid+BC ≈ ssj+BC at
n=400 and ~3% ahead at n=800 (1.49× vs 1.44× over plain ssj) — the plain
hybrid's sweep saving is visibly eaten by its per-block overheads (re-forming
B, re-estimating the norm each outer block). A quiet-box wall run is owed.

**What this sharpens for next ticks:** the question is no longer "when can
the manifold be left" (answer: at the cliff, worth ~20%) but **"what makes the
pre-cliff phase long, and can anything cheaper than manifold sweeps shorten
it"**. BC shortens it 2× by handing the iterate diagonal spread. The next
lever on the same axis: the cliff fires when the *sorted diagonal ordering
stabilizes* — worth instrumenting whether the pre-cliff sweeps are spent
sorting eigenvalue positions rather than resolving couplings.

153 + 2 tests pass (`test_hybrid_composes_with_block_preconditioner`,
`test_hybrid_falls_back_to_ssj_on_exact_ties`).

### 9. Anatomy of the pre-cliff phase — and the block-size schedule it implies

Attempt #8 left the question "what makes the pre-cliff phase long?" This tick
instrumented it per sweep (all load-immune, shipped one-sweep code path):
rel_off, contraction factor, ρ and the sorted-position distance of the pair
attaining it, diagonal accuracy vs true eigenvalues, spread ratio, and the
count of diagonal entries not yet within half a local gap of their eigenvalue.

**The anatomy (GOE n=400 plain, 23 sweeps):**

* **The binding pair is always adjacent** — sorted-position distance 1, every
  sweep, every case measured. The constraint is always a local resonance.
* **Assignment completes exactly at the cliff**: unassigned 378 → 13 → 0
  precisely as ρ crosses 1. The cliff *is* the last level crossing resolving.
* **No slow middle**: contraction improves monotonically 0.99 → 0.92 → 0.83 →
  0.66 → 0.38 → 0.09. One self-accelerating loop — couplings feed the
  diagonal, spread widens gaps, wider gaps shrink the saturated angles.
* **Cost concentrates at small spread**: sweeps 0–8 move rel_off 9.9 → 4.5
  (one decade); sweeps 15–23 move it thirteen decades. BC's mechanism is
  visible directly: one BC sweep takes spread 0.105 → 0.637 vs 0.260 plain.

**The lever: a per-sweep block-size schedule.** Attempt #4's grid only tested
*fixed* m (flat optimum, a recorded dead end). The anatomy says m should be
big exactly while spread is the bottleneck and small after. Measured, chained
first, then shipped in-solver (`block_m` now accepts a sequence, last entry
repeating; scalar path bit-identical):

| case | fixed m=32 | schedule [n/2, n/4, 32] |
|---|---|---|
| GOE n=400 | 11 | **7** |
| GOE n=800 | 15 | **8** |
| clustered 1e-9 n=200 | 9 | **7** (as [100,50,32]) |
| 5-fold deg n=200 | 25 | 27 — **no help, as the mechanism predicts** |

Degeneracy is bottlenecked on tie-resolution, not spread; the schedule is not
for it. Accuracy asserted on every configuration (≤6e-15 / ≤4e-14 residual).

**Wall — both runs CLEAN (3.2% / 4.1% contamination), the first fully clean
wall session of this log:**

| n=800 | ms | ×LAPACK | vs plain |
|---|---|---|---|
| plain ssj | 2600 | 33.9× | 1.00× |
| BC32 fixed | 1800 | 23.5× | 1.44× |
| BC schedule | 1368 | 17.8× | 1.90× |
| hybrid + BC schedule | **1219** | **15.9×** | **2.13×** |

(n=400: 1.95× and 2.06×, 13.5× LAPACK.) This also delivers the wall number
owed by #8: the IPT hand-off is worth a further ~11% on top of the schedule,
clean-measured. **The campaign's aggregate on a cold GOE solve now stands at
2.1×, taking SSJ from 28–34× LAPACK to 13.5–15.9×.**

**What the schedule is, said honestly:** [n/2, …] is one level of
divide-and-conquer with an SSJ merge — diagonalize two halves by batched
`eigh`, then let saturated sweeps merge them. The remaining cost IS the merge.
That reframing sets the next question (now in the reflection section): after
the halves are diagonalized, is the coupling effectively banded in sorted
order? If the binding pairs are only the ~n interleaving adjacencies, merge
sweeps could act on a band at O(n²·b), which is exactly the line that decides
whether an SSJ-style merge can ever compete with LAPACK's O(n²) secular merge.

159 tests pass (4 new: schedule wins on GOE, scalar/singleton bit-equivalence,
degeneracy guard, entry capping).

### 10. The merge is irreducibly global — locality is a red herring

Attempt #9 queued the decisive question for the divide-and-conquer reading:
after the halves are diagonalized, is the coupling banded in sorted order, so
merge sweeps could act on a band at O(n²·b)? Answer: **no, twice over**, and
the reason is the campaign's sharpest mechanism finding.

**The localization law is real — for the wrong quantity.** After the [n/2]
pass at n=400, cumulative squared-mass within sorted-position distance d:

| | d≤8 | d≤32 | d≤64 | d≤128 |
|---|---|---|---|---|
| angles K | 0.45 | 0.80 | **0.91** | 0.97 |
| couplings B | 0.04 | 0.15 | 0.30 | 0.54 |

K localizes (~1/d², summable — gaps grow linearly with sorted distance) but
**B does not**, and off(B) is what must die.

**Refutation 1 — block passes only.** A sorted-block pass is an exact
orthogonal similarity: B and X update incrementally, no B re-form, no angle
map, no retraction — the manifold is free. If rotation were local this would
be the whole merge. It stalls at rel_off 4.1–5.9 on *every* spectrum (GOE,
degenerate, clustered), at exactly the long-range B-mass the local passes can
never reach. (Side-finding worth keeping: after 160 incremental exact-
similarity passes, X's orthogonality drift is only 3e-13 — no-reform
bookkeeping is numerically sound.)

**Refutation 2 — banding the map itself.** Mask K to |i−j| ≤ b in sorted
order, keep everything else (full B re-form each sweep, dense QR), so dropped
pairs return every sweep. Even **b=128 — 97% of K-mass — does not converge**
(dlam 1e-1 at 120 sweeps, vs 24 sweeps full). b=8…64 all likewise.

**The mechanism:** annihilating a pair removes its *entire* coupling
regardless of the angle's size — work done scales with coupling mass, not
rotation mass. The O(n²) tiny far-pair rotations ARE most of the B-mass
removal. So the map's global reach joins the two saturations as load-bearing,
the dense retraction is the price of global annihilation (closing the
"cheapen QR via structure" line for good), and the elegant-looking banded
merge is dead with a clean cause of death.

No code shipped (scratchpad prototypes only; negative result). Where this
leaves the campaign is rewritten in the reflection section: sweep count near
floor, per-sweep cost floored — the unexploited axes are precision (the
memoryless map makes the expensive globalization phase noise-tolerant) and
hardware (everything shipped is GPU-shaped). Next tick: the full
mixed × schedule × hybrid composition, never measured together.

### 11. Mixed × schedule × hybrid — the composition, and why gains don't multiply

Queued by #10. Verify-before-believing found the predicted defect first: the
mixed branch forwarded the whole schedule to the fp64 phase, whose sweep
counter restarts — so it re-fired the n/2 and n/4 blocks as *fp64* batched
eigensolves on an already-warm iterate. Measured: identical sweep counts with
or without them — pure waste (~12 modelled gemm-equivalents at n=800).

**The policy that shipped: phases own schedule segments.** The fp32 phase
gets the full schedule (the spread problem is its job); the fp64 phase gets
only the schedule's small tail entry. The tail is not optional — structure
below fp32's ~1e-7 resolution is invisible to the low phase and arrives at
the fp64 phase unresolved, where small blocks are exactly what resolves it:

| case | fp64 phase w/ tail blocks | w/o |
|---|---|---|
| clustered 1e-9 n=400 | **2 sweeps** | 5 |
| 5-fold deg n=200 | **18** | 32 |

**Wall.** n=1200 CLEAN (1.0% contamination); n=800 at 11.8% (suggestive);
n=400 at 40.6% (discarded):

| n=1200 | ms | ×LAPACK | vs plain |
|---|---|---|---|
| plain ssj | 7068 | 41.6× | 1.00× |
| sched fp64 (#9) | 3398 | 20.0× | 2.08× |
| sched mixed | 3571 | 21.0× | 1.98× |
| hybrid sched fp64 | 3297 | 19.4× | 2.14× |
| **hybrid sched mixed** | **3038** | **17.9×** | **2.33×** |

(n=800 suggestive: 2.35×, 16.3× LAPACK.) Load-immune: 7 fp32 + 2 fp64 sweeps
replace 9 fp64 at n=800; accuracy ≤ 8.2e-15 everywhere, asserted.

**The finding worth keeping: composition is Amdahl, not multiplication.**
Schedule 1.90×, hybrid 1.44× alone (#8-era), mixed 1.35× alone (historical) —
"expected" product 3.7×; measured 2.33×. All three shorten the *same*
expensive globalization phase, so each one shrinks what the next can save.
Note also `sched mixed` *alone* loses to `sched fp64` at n=1200: fp32 batched
`eigh` is only ~1.3× faster than fp64 (unlike sgemm's 2×), and once big-block
eighs are a large share of the remaining cost, mixed's margin dies — it only
pays (~8%) inside the hybrid, which replaces the fp64 tail with IPT and
leaves proportionally more of the solve in sgemm-land.

**Aggregate standing:** cold GOE CPU solve 16–18× LAPACK, from 28–42× at
campaign start. This axis is near its floor; the reflection now points the
next ticks at the GPU notebook (predates schedule/hybrid/mixed-policy) and
the tracking niche.

160 tests pass (1 new, pinning the fp64-phase tail-block policy on the
fp32-invisible cluster).

### 12. The GPU notebook now carries the composed solver

The reflection has called the GPU run the highest-value open experiment since
#10, but `ssj_gpu_colab.ipynb` predated everything the campaign shipped: it
still had the #2-era unconditional Newton–Schulz floor (measured in #5 as a
1.60× penalty), the old temporaries-heavy `_angles`, no schedule, no IPT, no
hybrid. A GPU run of that notebook would have measured a solver two-and-a-half
generations stale.

Ported into the notebook's standalone solver, faithful to core through #11:
per-sweep block schedules; the predictive NS tightening (margin 1e9, with the
comment explaining why the always-on floor was a 1.60× loss); the in-place
`_angles` arithmetic (keeping the notebook's documented GPU deviation — the
tie branch runs unconditionally rather than paying a host sync); the mixed
phases-own-schedule-segments policy; and a new cell with `ipt_rate`,
`ipt_iterate` (pure gemm, in-place elementwise) and the `ssj_ipt` hybrid
including the clean-hand-off QR from #8. Benchmark cells now run six configs
(plain, BC32, schedule, mixed+schedule, hybrid, hybrid+mixed) against
cuSOLVER, with the CPU control and the ratio-shift verdict joined on shared
labels, and the framing text updated to the campaign's measured standing.

**Validated the way the original was — by running the notebook's own cells on
the NumPy path** (the solver is backend-agnostic): 50 config × spectrum rows
all pass with accuracy asserted. Notable rows: hybrid+sched solves GOE n=200
in **4 sweeps + 8 IPT gemms** (plain: 21 sweeps) and zero-diagonal in 4 + 6;
exact degeneracy correctly never opens the gate and falls back to SSJ+blocks.
The one honest caveat surfaced again: on the 1e-9 cluster the mixed hybrid's
eigenvector orthogonality is 4.5e-10 (IPT vectors are only implicitly
orthogonal; eigenvalues and residuals at full precision) — the validation
bar encodes that documented trade rather than hiding it. Driver cells run
end-to-end under a NumPy alias with the verdict join matching 6/6 rows.

No solver-repo code changed; this tick ships measurement infrastructure. The
GPU question is now fully instrumented and waits only for a card.

### 13. The floor audit — what "floor" actually meant

Prompted by a direct challenge to #11's "near its floor" claim. The claim
conflated an *algorithm* gap with a *substrate* gap, and this tick measured
the decomposition.

**The corrected accounting.** In flop units (one n³-flop-pair gemm = 1.0) the
composed solver costs ~30–40 gemm-equivalents against LAPACK `dsyevd`'s 8–18:
**the algorithm is within ~2–3× of LAPACK.** The wall ratio of 16–18× carries
an additional ~6× of substrate, decomposed per component (n=800, quiet start;
the n=1200 rows of this run are load-suspect and excluded):

| component | flops | measured | overhead |
|---|---|---|---|
| gram `X.T @ X` (numpy) | 1.00 | 0.98 | **1.0×** |
| B-form (2 gemms + symmetrize) | 2.00 | 3.88 | 1.9× |
| `_orth_qr` (LAPACK dgeqrf/dorgqr) | 2.67 | 8.94 | **3.3×** |
| shipped `_orth_cholqr2` | 6.00 | 26.69 | 4.4× |
| cholqr2 rebuilt on BLAS (syrk+potrf+trtri) | ~3.7 | 17.95 | 4.9× |

numpy's gemm and Gram are AT the flop floor — nothing to reclaim there. The
big overheads are LAPACK QR's panel serial fraction (unreachable from Python)
and the B-form's symmetrize+temporaries (~1 ge/sweep, fused-C territory).

**The symmetric-BLAS hypothesis failed, and the failure was diagnostic.**
dsyrk "should" cost half a gemm; measured, it cost 1.7× numpy's full gemm.
The control that explained it: **scipy's `dgemm` itself runs 2.57× slower
than numpy's `@` on this box** — the two link differently-configured BLAS.
Within scipy's own library, dsyrk is 0.58× dgemm, i.e. the half-flops promise
is real — but this stack's only fast BLAS entry point (numpy's `@`) exposes
no symmetric kernels, so the saving is unreachable from Python here. Without
that control, the table above would have "refuted" syrk as a kernel; it
actually measured a packaging accident. Third instance of the same method
lesson: an unexamined identity assumption (scipy BLAS = numpy BLAS) invisibly
contaminating a comparison.

**What this settles and un-settles:**

* Settled: on this box, in numpy, the shipped implementation is close to what
  the substrate permits; further CPU wall gains are not sitting in kernel
  choice. #11's claim survives *as scoped*: the floor of this algorithm, in
  this substrate.
* Un-settled (and now stated in the reflection): the 6× substrate is mostly
  an artifact of this stack. On a gemm-dominant substrate the flop-unit
  picture governs — the algorithm at 2–3× LAPACK, CholeskyQR2's #7 verdict
  plausibly flipping (its flop content is 6.0 all-gemm vs QR's 2.67
  factorization-shaped) — which is precisely the notebook's question.
* Also un-settled: "deferred orthonormalization: no gain" dates from the
  25-cheap-sweeps era. At 8 sweeps with QR at ~50% of each, the economics may
  invert. Queued as an era-stale dead-end worth one re-measurement.

No solver code changed; scratchpad only. The headline correction — the
campaign's honest standing is *2–3× LAPACK in flop units*, with the rest
substrate — is now in the reflection where it can steer.

### 14. Tracking — the niche dies on CPU, and two warm-path defects die with it

Queued since #11. Three results: two shipped fixes, one standing claim killed.

**Defect 1 — the schedule fired its head on warm starts.** The schedule
indexes by sweep count, so a tight `X0` paid an n/2 + n/4 batched eigh per
early sweep for spread it already had: same sweep counts, 1.19× wall (same
defect shape as #11's mixed-phase fix). At ε=1e-2 the head did save one sweep
(4 vs 5) — but the eigh it costs per sweep exceeds the sweep it saves, so
skipping is right even there. **Fix: state-based, not X0-based** — the
schedule jumps to its tail entry whenever rel_off < 0.3 (`_SCHED_HEAD_GATE`),
using a quantity the loop already computes. A junk X0 still fires the head;
cold solves are bit-identical with and without the gate (verified: histories
equal at gate 0.3 vs 0.0).

**Defect 2 — the entry QR re-orthonormalized an orthonormal basis.**
`_orth_qr(X0)` ran unconditionally: 12 gemm-equivalents (47 ms at n=800), the
single largest item in a warm solve, spent un-rotating a previous eigenbasis
that is orthonormal to 1e-14. Now a 1-ge Gram check decides; any defect below
the threshold is corrected at the first retraction anyway (the product form
re-measures X's defect — the module's own load-bearing property). The mixed
path is unaffected: its fp32→fp64 hand-off has a ~1e-7 defect and still fires
the QR, as it must.

**The crossover chart — clean box, 1.7% contamination, accuracy asserted:**

| ε (n=800) | warm ms | sweeps | LAPACK ms | warm/LAPACK |
|---|---|---|---|---|
| 1e-8 | 126.5* | 1 | 48.3 | **2.6×** |
| 1e-4 | 291.2* | 3 | 48.9 | **6.0×** |
| 1e-2 | 486.3 | 5 | 48.0 | 10.1× |
| 1e-1 | 502.2 | 5 | 47.9 | 10.5× |

(*after both fixes; pre-fix 1e-8 was 161.9.) **Warm never beats LAPACK at any
ε.** The mechanism is arithmetic, not implementation: one SSJ sweep measures
~14.5 gemm-equivalents while LAPACK's entire `dsyevd` is ~13.8 on this box —
when a single sweep costs more than the incumbent's full solve, no warm start
can win, however perfect the basis. The long-standing "warm start: 1–5 sweeps
beats re-solving" claim compared SSJ-warm against SSJ-cold and never against
the incumbent. **The tracking niche is a property of the substrate**: it
exists exactly where the incumbent is expensive relative to a gemm — cuSOLVER
at 30–40 gemm-equivalents — which the GPU notebook's warm-tracking cell now
decides. (The two warm fixes belong in the notebook at its next sync.)

**Also caught: a phantom regression from cross-era numbers.** The head-gate
check "expected 8 sweeps at n=800, got 9" — the 8 was from #9's *chained*
prototype (which re-orthonormalizes between one-sweep calls), never the
in-solver path. Gate on/off A-B showed bit-identical behaviour. Fourth
instance of the measurement-lesson family: never compare against a number
from a different code era without re-measuring it.

162 tests pass (2 new: entry-QR guard accuracy both branches; head-gate
bit-equivalence with tail-only on warm starts).
