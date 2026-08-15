# SSJ improvement log

A dedicated track, separate from `MAP_LEDGER.md`. That ledger hunts for *new
maps*; this one improves the map the repository already ships. **Append here
before anything else**, one line per attempt, negative results included.

## What this is teaching us about the eigenvalue problem

*(rewritten each tick; as of attempt #8)*

**Fast eigensolvers divide by gaps.** Every method measured fast here takes
steps of the form coupling/gap — Jacobi angles, IPT denominators, secular
solves. Every method that doesn't (gradient/Brockett ~800×, homotopy, plain
flows) is slow. Dividing by the gap is a Newton step in disguise: it uses
curvature. Its price is that it explodes near resonance (gap → 0), so it only
works inside a basin (ρ = max|W|/gap < 1) or behind a saturation.

**The two-phase decomposition is the deep structure.** Every practical solver
= a *globalization* phase that buys the basin + a *divide-by-gap endgame* that
exploits it. LAPACK: tridiagonalization, then gap-divided secular/QL steps.
SSJ: saturated sweeps, then the same sweeps acting quadratically. The phases
have different economics and the mistake this repo's cost discussions kept
making was pricing them as one thing.

**What the manifold is actually for.** SSJ's orthogonality constraint (QR,
49% of every sweep — attempt #5) is not decoration: both of its saturations
are load-bearing (arctan on angles; polar on the composed step — Cayley's
weaker saturation diverges at n=800, attempt #7). But the saturations are only
*needed outside the basin*. IPT proves the endgame needs no manifold at all:
pin v_jj = 1, iterate one gemm per step, done. So the manifold tax is a
globalization cost being paid during the endgame too, where it buys nothing.

**Corollary being tested now (attempt #8):** the right hybrid is
globalize-cheaply-then-flee-the-manifold. The repo already ships the pieces —
`ssj_ipt_eigh` (handoff gated on ρ < 0.5) and SSJ-BC (a globalizer: block
passes hand the iterate √m of diagonal spread) — but they predate each other
and have never been composed or measured together.

**Deepest open question:** what is the *cheapest sufficient* globalization?
The basin condition is ρ < 1, an O(n²) observable. SSJ buys it with O(n³)
manifold sweeps; BC buys spread with batched small eigensolves. Whether
something O(n²)-per-step can buy it (or whether the gate opens after just 2–3
BC sweeps) decides how much of the n³ manifold work is actually necessary.

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
