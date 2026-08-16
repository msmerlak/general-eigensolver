# SSJ improvement log

A dedicated track, separate from `MAP_LEDGER.md`. That ledger hunts for *new
maps*; this one improves the map the repository already ships. **Append here
before anything else**, one line per attempt, negative results included.

## What this is teaching us about the eigenvalue problem

*(rewritten each tick; as of attempt #28)*

**The canonical structure, arrived at and then validated by refutation
(#20): every solver here is a COARSE SUPPLIER plus the REFINEMENT LADDER.**
The ladder (consult-A IPT step ⟂ Newton–Schulz step, alternating — skip
either and you floor, #19) squares the error per ~5-gemm pair. Its basin was
*measured, and it is small*: convergence from coarse error ≲ 1e-3..1e-4,
proportional stall beyond (0.2× the corruption from 1e-2). So refinement
buys the last 7–11 digits and never the first four. The first four are the
actual eigenvalue problem — and they are precision-free, which is where all
the substrate leverage lives.

**Everything measured this campaign slots into that frame.** SSJ = a coarse
supplier with a global-rotation mechanism (its own endgame was already the
ladder's ancestor). Purification = a coarse supplier with a global basin and
pure-gemm flops (fp32-split champion #19 IS coarse+ladder). LAPACK dsyevd =
a coarse supplier that happens to go all the way. Tracking = a coarse
supplier from history. `refine_eigh` now ships the ladder as a public API:
upgrade ANY ≳1e-4 basis — a GPU fp16/fp32 eigensolve, a tracked basis,
either family's split — to fp64 at ~5 gemms per squared digit, no
factorization.

**Optimization is component profiling plus mechanism-driven guards (#21).**
The batch: Gershgorin was 12.5× loose on GOE and the SP2 seed slope is
inversely proportional to the enclosure width — tight power-on-A² bounds
(inflated 25%, Gershgorin-clipped) cut the linear phase; SP2's loop ran in
preallocated ping-pong buffers (the P←P² branch is a pointer swap); the
polish went in-place (|gap| < 10³|W| ⟺ |C| > 10⁻³, one mask absorbing
inf and NaN); the trailing NS step was dropped (final defect = err² ≈
1e-16); G is cached. Two robustness mechanisms surfaced and shipped: a
power-estimate is NOT an enclosure — under-enclosed eigenvalues survive the
McWeeny warmup (its attractor basin reaches ~1.37) and explode only later
under SP2's squaring branch, so the divergence guard lives in the periodic
check and retries from the exact Gershgorin enclosure; and the fp32 route
can cut *through* a 1e-9 cluster it cannot see, caught by a free post-hoc
boundary audit (the boundary gap is known after recursion) with an `eigh`
fallback. **Champion now 4.4×/5.3×/6.7× LAPACK at n=400/800/1600, residuals
1.3e-15–2.0e-13.**

**The champion has no unnecessary steps on this substrate (#22).** Four
existence-level cuts tried, three refuted with mechanisms: the all-fp32
split pipeline (sgeqrf runs at dgeqrf speed — ambush #7 — so fp32 QR saves
nothing and the casts cost, while accuracy drops); inertia-exact rank via
sytrf replacing the McWeeny warmup (ssytrf costs as much as the warmup it
replaces); a conditional second polish (its cheap signal — orthogonality
defect — is NOT a residual estimate; skipping by it broke the battery). One
survived at noise level (bounds 15 → 7 power iterations). The crisp law
underneath: **on this stack, reduced precision pays only through gemm** —
sgemm is 2× dgemm, but ssyevd, sgeqrf and ssytrf all run at fp64 speed —
which is exactly why the SP2 phase (pure gemm) is the only place mixed
precision ever bought anything here, and why tensor-core substrates change
every one of these verdicts at once.

**The compiled port confirms the substrate thesis, partially (#23).** A C
implementation linking the SAME OpenBLAS lands **3.3×/4.3×/5.2× dsyevd**
against Python's 4.2×/5.3×/6.7× — a 1.3× implementation gain from `dsymm`
(half-flop symmetric products NumPy's `@` cannot express), `dsyrk`, one
fused pass where NumPy needs six, and no allocation in any loop. But it does
NOT reach #13's flop-model prediction of 2–3×, and the reason is now visible:
the ratio *grows with n* (3.3 → 5.2 over a 4× size range) because dsyevd's
own leaves and the O(n³) split both scale the same, while dsyevd's constant
improves with size. The algorithm's flop deficit is real, not an artifact —
#13 was right that ~1.3× was substrate and wrong that all of it was.

**The non-symmetric case: the architecture transfers, the splitter does not
(#24).** Purification is a REAL-LINE method — its seed maps lambda to
0.5 + c(mu - lambda) and the map's attracting basins are bounded regions of
C, so it diverges on complex pairs (real Ginibre, |Im|/|lambda| = 0.98) and
on real spectra once eigenvectors are ill-conditioned. The matrix SIGN
function is the correct substitute in the same architectural slot (gemm-only,
global basin, splits on the line Re(z) = sigma) and is already in the repo.
The split then becomes block-TRIANGULAR rather than block-diagonal, and its
backward error is governed by ||P||, the OBLIQUE projector norm: measured
||P|| 3.2 -> 1.5e5 and ||A21||/||A|| 1.5e-14 -> 1.6e-9 as cond(X) runs
10 -> 1e6. **For symmetric A that norm is identically 1** — which is exactly
why the symmetric method is unconditionally stable and the general one is
not. The ladder loses its Newton-Schulz half (non-symmetric eigenvectors are
not orthonormal) but the consult-A half works: 1e-8 -> 2.8e-16 in 4 IPT
iterations. And the prize is ~7x larger: dgeev costs 131-181 gemm-equivalents
here against dsyevd's 17-25.

**The non-symmetric race, run (#25): parity in operations, 5.3x loss in
wall.** SDC-by-sign against dgeev on Ginibre n=400: **88 gemm-equivalents
counted against dgeev's 89 — parity** — but 407 ms against 76 ms measured.
The architecture is not the problem; this implementation of it is, and the
cost is localized: **one `matrix_sign` call is 6.3x the entire dgeev solve**,
and sign + leaves are 90% of the run. Where inside `matrix_sign` remains
unattributed (denormals refuted; my per-iteration decomposition was
single-shot and cache-cold, so it was discarded rather than reported). So
`sdc.py`'s founding claim — a method doing several times more arithmetic
still wins if the arithmetic is gemms — is **true in the ledger and false at
the wall**, pending that one function. The leaf lesson of #17 did transfer:
one split (leaf = n/2) beats recursing to 2x2 by **4x** with equal accuracy,
now the shipped default.

**A gate that is a MAX is a statistic, not a property of the matrix (#26).**
IPT's admission test is `rho = max` over ALL pairs, so one tight k-cluster
disqualifies a whole matrix whose other n − k columns sit at `rho_j ~ 1e-3`.
Because the IPT map is column-separable, restricting it to the admissible
columns is not deflation or locking — it is the same iteration on fewer
columns, exact by construction — and the resonant remainder goes to a dense
solve on a deflated |C|-dimensional basis. Measured 1.3e-15–8.6e-15 where
plain `ipt_eigh` fails at 7e-07–1e-04. But the speed is **parity** (0.97–1.08×
dsyevd), and structurally so: dropping k of n columns saves k/n of IPT's cost.
**The column split buys ADMISSION, not throughput.** That generalizes past IPT
— every basin-limited method here is gated by an aggregate, and the aggregate
is almost always a max over pairs.

**And separability cuts both ways, which is the sharper half.** The property
that makes the restriction exact is the same one that lets two columns
converge to the SAME eigenpair, since nothing couples them: measured at
n=1600, 1582 columns flagged *converged* with rank 1581, two of them returning
the identical eigenvector to |⟨v_j,v_p⟩| = 1.000000. The flags are honest —
each vector really is a fixed point of its own column's map. **Per-column
convergence is a statement about one column's residual and certifies nothing
about the basis being complete**, so the cheap `rho_j` screen is not an
optimization that could be replaced by the solver's own convergence flag; it
is the only thing standing between a column-separable method and a silently
rank-deficient answer. Same shape as #15's verdict that off(B) in a skewed
frame certifies nothing: *a local residual is not a global certificate.*

**The compiled SDC settles #25's paradox: the ledger was right, NumPy was
the gap (#27).** A C port on the same OpenBLAS takes SDC from 7.7–14× slower
than dgeev to **1.6–1.8× slower** (0.56×/0.61×/0.61× at n=200/400/800) — a
**5–8× implementation gain**, against the symmetric port's 1.3× (#23). So
`sdc.py`'s founding claim — a method doing several times more arithmetic
still wins when the arithmetic is gemms — was true in the operation ledger
and false at the wall *only because of the implementation*, and the wall has
now moved most of the way to meet the ledger. The non-symmetric side was far
more substrate-bound than the symmetric one, which is itself the lesson: the
more a method leans on kernels NumPy cannot express (here `dgetri`, and
fused passes over n² that NumPy pays as separate temporaries **every**
iteration), the larger the compiled gain. Phase attribution, now stable
across three sizes: sign 66–71%, leaves 15–19%, pivoted QR 13–14%, gemms
2.6%. **The named remainder, now fixed (#28): two sign evaluations per solve
where one was needed** — not a rejected shift, as #27 guessed, but a leaf of
exactly n/2 sitting just under `r = trace(P)`. Leaf 3n/5 takes the C solver
to **0.60×/0.67×/0.65× dgeev** and the Python one from 412 to 207 ms at
n=200. What remains is the sign function itself: one evaluation costs
0.77–1.02× an entire dgeev solve, and the flop model says the opening is the
Newton/Newton–Schulz switch, since both steps cost 4n³ but only NS is pure
gemm.

**Substrate ambushes, now six of a kind:** ssyevd runs at dsyevd speed on
this box (1.01×/0.96× — the fp32-LAPACK-coarse idea dies HERE and is the
natural headline on tensor-core substrates). The campaign's portable output
is mechanisms and structure; every ×-number is a claim about one substrate.

**Where this leaves the competition on this CPU:** dsyevd is its own coarse
supplier at 1×, so nothing beats it locally except by supplying coarse more
cheaply than LAPACK — purification-mixed at 4.9–6.7× is the best
LAPACK-free answer. The open decisive question is unchanged and now sharper:
**on substrates where low-precision is genuinely cheap (GPU), coarse+ladder
is the design, and the notebook should race it** — fp16/fp32 cuSOLVER or
fp32-SP2 as coarse, ladder to fp64. Then the SSJ convergence proof.

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
| deferred orthonormalization | **actively harmful** (#15, re-measured at 8-sweep economics): every skip rule costs sweeps, and it can terminate "converged" with 1e-7 eigenvalue error — off(B) in a skewed frame certifies nothing |
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
| 15 | **Re-measure the era-stale dead end**: deferred orthonormalization at 8-sweep economics | **dead end confirmed and upgraded** — it breaks the convergence certificate, not just the speed. The retraction is also the stopping test's meaning. |
| 16 | **Is IPT the only pure-gemm endgame?** (user steer) — map the families; prove ρ(p(B)) invariance; measure the purification recursion end to end | **basin invariance proven + verified; purification loses 1.3× on CPU at full accuracy — the strong second family, GPU-shaped.** |
| 17 | **Close purification's gap**: randomized extraction, leaf tuning, mixed purify | **NEW CPU CHAMPION — 8.3×/9.9× LAPACK** (vs composed SSJ 12×/16×), shipped as `purify_eigh` + tests. Mixed purify refuted with mechanism: the map freezes subspace error. |
| 18 | **Compose the families**: one IPT step polishes the purified basis; SP2 replaces McWeeny; split-verification fallback | **champion at full accuracy: 6.0–9.3× LAPACK, resid 3e-15.** Suite caught SP2×degeneracy; safety net free on the happy path. |
| 19 | **fp32 splits + the refinement ladder** (consult-A polish ⟂ NS re-orth, alternating) | **champion again: 4.9×/6.7× LAPACK shipped as `precision="mixed"`.** The polish alone floors at err² — the interleaved NS step restores quadratic refinement. |
| 20 | **Rethink the structure**: is the ladder the whole algorithm? Measure its basin; test fp32-LAPACK coarse | **structure settled: coarse (≲1e-4, precision-free) + ladder (last digits). Basin is SMALL (stalls from 1e-2), ssyevd = dsyevd here (ambush #6). `refine_eigh` shipped as the reusable half.** |
| 21 | **Optimize the champion**: profile-driven batch (tight bounds, in-place SP2, in-place polish, drop trailing NS, cached G) + two new guards | **4.4×/5.3×/6.7× LAPACK at n=400/800/1600 (from 4.9×/6.7×), all clean runs, full accuracy.** |
| 22 | **Cut unnecessary steps**: all-fp32 pipeline, inertia rank, conditional polish, bounds trim | **minimality verdict — three cuts refuted with mechanisms, one at noise level. Every surviving step earns its place; CPU optimization is converged.** |
| 23 | **Compiled implementation** (C + the same OpenBLAS): `csrc/purify_eigh.c` | **3.3×/4.3×/5.2× dsyevd at n=400/800/1600**, from Python's 4.2/5.3/6.7 — 1.3× is implementation, and the flop deficit is confirmed real. |
| 24 | **Does the method transfer to non-symmetric A?** (user question) — splitter, split quality, ladder, economics | **architecture yes, splitter no.** Purification is real-line-only; sign replaces it. Split error scales with the oblique ||P||. Incumbent is 7x weaker — the bigger prize. |
| 25 | **Race SDC-by-sign against dgeev** end to end; leaf sweep | **parity in op counts (88 vs 89 ge), 5.3x loss in wall.** Leaf lesson transfers (4x, shipped as default). The gap is one function: `matrix_sign`. |
| 28 | **Improve the C SDC** — instrument the shift guards, then the leaf | **the #27 diagnosis was a guess and was wrong: ZERO shifts are ever rejected.** The second sign call is a second split, caused by leaf = n/2 landing just under r = trace(P). Leaf 3n/5: **0.60×/0.67×/0.65× dgeev** (C), 412→207 / 1315→748 ms (Python). |
| 27 | **SDC in a compiled language** (user request) — `csrc/sdc_eig.c`, same OpenBLAS | **0.56×/0.61×/0.61× dgeev at n=200/400/800, from Python's 0.07×/0.13× — a 5–8× implementation gain.** #25's ledger was right; NumPy was the whole gap. Sign is 66–71%, and one wasted shift retry is ~1/3 of the run. |
| 26 | **Can IPT run on the separated eigenvalues while another algorithm takes the rest?** (user question) — column split, shipped as `ipt_hybrid_eigh` | **yes, and exactly: 1.3e-15–8.6e-15 where plain IPT fails at 7e-07–1e-04.** Speed is parity (0.97–1.08×) — the split buys ADMISSION, not throughput. Screen is a SAFETY property: unscreened, two columns converge to the *same* eigenpair. `ipt_rate_columns` vectorized, bit-identical, 2.4–4.6×. |

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

### 15. Deferred orthonormalization — the economics did not invert

The dead-end entry "deferred orthonormalization: no gain" dated from the
25-cheap-sweeps era; with QR now ~50% of each of ~8 sweeps, skipping two
retractions would be worth ~18 gemm-equivalents, so the entry was re-measured
rather than trusted (the era-stale concern raised in #13).

Harness: the shipped auto path (schedule + head gate + NS floor + endgame)
with a pluggable skip rule, and — the #14 lesson — an internal baseline in the
same harness, which reproduces shipped `ssj_eigh` sweep counts exactly
(7/9/7). Skipped sweeps run `X ← X(I+K)` unretracted; the run ends with one
exact QR. All load-immune; accuracy asserted.

| rule (GOE n=400) | sweeps | dlam | resid | verdict |
|---|---|---|---|---|
| none (baseline) | 7 | 4.0e-15 | 3.3e-15 | — |
| skip when ‖K‖_F < 0.5 | 14 | **8.3e-07** | 6.8e-05 | FAIL |
| skip when ‖K‖_F < 0.1 | 8 | 3.9e-14 | **2.7e-08** | FAIL |
| skip alternate QRs (big-K phase) | 80 (cap) | 2.3e-02 | — | diverges |

(n=800 and the 1e-9 cluster: same shape; the cluster stays accurate but still
pays 7 → 12 sweeps.)

**Two mechanisms, one of them new to this log:**

1. *The skewed frame slows the map* — every rule cost sweeps even where
   accuracy survived. The angles computed from a congruence B are
   systematically poorer than similarity angles.
2. **The retraction is also the certificate.** The failing runs *terminated
   on the convergence test*: orthogonality was fine after the final QR
   (2.4e-14) but eigenvalues were wrong by 1e-7, because `off(B) ≤ tol` was
   evaluated in a skewed frame where off(B) is a congruence residual and
   bounds nothing about the similarity. The final QR restores the manifold
   but cannot retroactively restore the meaning of the test that already
   fired. Any future skip-the-retraction idea must supply a frame-independent
   stopping criterion first.

Dead-end entry upgraded from "no gain" to "actively harmful", with numbers
and the certificate mechanism. No code shipped; the retraction stands as the
price of three things at once — the manifold, the saturation geometry, and
the termination test's meaning. With this, every engineering item measurable
on this box is closed or shipped; the open list is the GPU run (one small
notebook sync pending) and the convergence proof.

### 16. The other pure-gemm family — and a small invariance theorem

Steered by a direct question: is IPT really the only gemm-only iteration
whose fixed points are eigenvectors? Answer: no — there are exactly two
families in play, plus a bridge between them that a short argument closes.

**The map of the space.** (1) *Divide-by-gap fixed points on the basis*:
IPT and Brillouin–Wigner (both shipped) — one gemm per iteration, linear at
rate ρ, basin ρ < 1, column-separable. (2) *Polynomial flows on the matrix*:
McWeeny purification P ← 3P² − 2P³ / matrix sign (both shipped, ledger
#20–21, but only ever measured against `dgeev` on the nonsymmetric side) —
two gemms per iteration, **quadratic, global basin** (the initial scaling
traps the spectrum in [0,1]), delivering an invariant-subspace split per run,
recursed to a full decomposition. (3) A structural relative, LU-normalized
subspace iteration (treppeniteration): pins a triangle instead of pinning
v_jj or keeping a manifold — but converges at eigenvalue-modulus ratios,
hopeless for GOE-like spectra, and its LU is a factorization anyway. Not
pursued.

**The bridge is closed by an invariance.** Running IPT on p(B) (any
polynomial — gemm-only, eigenvector-preserving) looks like a free basin
widener. It is not: for B = D + W, Daleckii–Krein gives
(p(B))_ij ≈ p[d_i,d_j]·W_ij, and the gap moves by exactly the same divided
difference, |p(d_j) − p(d_i)| = |p[d_i,d_j]|·gap_ij. **The factor that
transports the coupling is the factor that transports the gap**, so
ρ(p(B)) = ρ(B) + O(W²) for every analytic p. Verified on real SSJ-trajectory
frames: ratios exactly 1.000 at small W for shifted-square, shifted-cube and
Chebyshev T₃; erratic (0.19–300, frame-dependent, unusable) at large W; and
the predicted failure mode appears on cue — pairs with p′ ≈ 0 between them
(d_i + d_j ≈ 2c under squaring) lose gap and coupling together and O(W²)
blows ρ up 179×. This joins "ρ(J) is a diagonal-similarity invariant" in the
mechanism list: **the basin can be bought by rotation (SSJ sweeps), never by
spectral surgery.**

**The purification recursion, end to end** (prototype: split at trace/n,
recurse to leaf 64, leaf = `eigh`, vectors by two column-block gemms per
level; accuracy asserted):

| | purify-recursion | composed SSJ | LAPACK |
|---|---|---|---|
| n=800 wall | 1498 ms (21.7×) | **1139 ms (16.5×)** | 69 ms |
| accuracy | 5.4e-15 / ortho 7.9e-14 | 5.8e-15 | — |

Degeneracy and 1e-9 clusters come out clean (leaves absorb them). **The
composed SSJ keeps the CPU crown by 1.3×** — but the margin is small for a
first prototype, and the loser's fat is identifiable: 15 pivoted QRs (dgeqp3;
replaceable by a randomized range-finder, i.e. two more gemms) and per-call
overhead at small recursion levels. A model-vs-measure correction for the
record: my per-split extrapolation ("~79 gemm-equivalents") conflated gemm
*count* with size-weighted cost — the recursion issues 798 gemm calls whose
small members are overhead-bound; size-weighted the model was right, wall is
what decides.

**Why this matters beyond the scoreboard:** purification is the one solver
family here whose every flop is a full-rate gemm with a *global* basin — no
manifold, no retraction, no gate, no certificate subtlety (the projector IS
the certificate: trace and idempotency are frame-independent). On the GPU it
is the natural competitor to both cuSOLVER and composed SSJ, and it is now a
candidate cell for the notebook. No core code shipped this tick; prototype
in scratchpad.

### 17. Purification takes the CPU crown

Attempt #16 left purification 1.3× behind with "identifiable fat". Three
levers tested; two delivered, one refuted with a mechanism worth more than
the speedup it denied.

**Randomized extraction replaces the pivoted QR.** P is idempotent to tol,
so `QR([P·G₁, (I−P)·G₂])` (one unpivoted QR + two gemms, ~11.2 measured ge)
yields the exact split basis dgeqp3 was providing at 15.7–23.4 ge. Split
residual 3.2e-14.

**The leaf sweep: recursion depth was the overhead.** Levels below ~200 are
call-bound while `eigh(200)` costs 4.6 ms. leaf = n/2 — one bisection, two
LAPACK leaves — wins at both sizes. (This is the same LAPACK reliance the SSJ
schedule has: its block passes are batched `eigh`. Fair race.)

**Mixed purification is impossible, and the reason is structural.** The
diagnostic: mixed converges to a *perfect* projector (‖P²−P‖ = 2.3e-14) that
commutes with A only to 9.5e-6 — an exact projector onto the wrong subspace.
The map P ← 3P²−2P³ never consults A after the seed; **every projector is a
fixed point, so subspace error introduced at fp32 is frozen forever.**
Contrast the reflection's standing fact about SSJ: memoryless in A, so any
frame noise is re-measured away. Purification is memoryless in everything
except P. A cure would have to consult A again (one subspace-iteration step
+ Rayleigh–Ritz polish) — noted as the residual-polish lever, not attempted.

**The race — both runs clean (1.5% / 5.5% contamination), accuracy asserted:**

| | n=400 | n=800 |
|---|---|---|
| purify leaf=n/2 | **153.7 ms = 8.3×** | **690.0 ms = 9.9×** |
| purify leaf=200 | 153.7 ms | 813.7 ms |
| composed SSJ (16-attempt champion) | 220.8 ms = 12.0× | 1116.6 ms = 16.0× |
| LAPACK | 18.5 ms | 69.9 ms |

**First single-digit ×LAPACK of the campaign, by 1.44×/1.62× over the
incumbent.** Eigenvalues 5.9e-15; residuals ~3e-11 (split-boundary mixing,
documented); degeneracy and 1e-9 clusters clean via the leaves.

Shipped as `ssj.purify_eigh` (deterministic by default, LAPACK fallback when
the spectrum refuses to split at the mean), 165 tests pass (3 new). The
notebook sync — #14's warm fixes plus a `purify_eigh` cell — is now the last
engineering item before the GPU run decides all three contenders at once.

### 18. The families compose: IPT polishes purification, SP2 cuts its cost

#17 left two open flanks: residuals at ~1e-11 and the McWeeny split cost.
Both closed, and the closure is the campaign's thesis in miniature — each
family's weakness is exactly the other's strength.

**The IPT polish.** Purification's structural flaw (#17): the map never
consults A after the seed, so split-boundary subspace mixing survives to the
answer. IPT is *nothing but* consulting A — and in the purified basis
ρ ≈ 1e-11 ≪ 1, deep inside its basin. One step, 3 gemms + elementwise, with
a guard zeroing corrections whose denominator is under 1e3× the coupling
(inside clusters the subspace is already invariant):

| spectrum | resid before | after |
|---|---|---|
| GOE n=800 | 2.4e-13 | **3.1e-15** |
| clustered 1e-9 n=400 | 1.5e-13 | 2.9e-15 |
| 5-fold deg n=200 | 1.4e-13 | 2.8e-15 |

Full LAPACK-grade accuracy; the #17 caveat is gone.

**SP2 replaces McWeeny.** Niklasson's trace-branched squaring (P ← P² or
2P − P²) does one gemm + a trace per iteration against McWeeny's two gemms
plus temporaries. Gemm counts barely move (55 vs 59) — **the 1.3–1.4× wall
win is substrate, not flops**, consistent with #13: fewer n² temporaries per
gemm issued.

**The suite caught what the tick script missed.** SP2 × exact degeneracy
broke (resid 0.30): eigenvalues exactly at μ sit at the purification fixed
point ½, and the trace branch mis-ranks, cutting inside a degenerate
cluster. My race script never ran that combination — `test_purify_eigh_hard_
spectra` did. Fix: a split-verification fallback, free on the happy path (B
is already formed — check ‖B₂₁‖ and fall back to `eigh` on a bad split).
Fifth member of the measurement-lesson family: a new code path needs the
full spectrum battery, not the benchmark's spectra.

**Shipped champion, verified post-ship** (n=800 clean at 1.6%; n=400 row
contaminated at 14.6% and quoted from the earlier clean run):

| | n=400 | n=800 |
|---|---|---|
| `purify_eigh` (SP2 + polish) | **87.7 ms = 6.2×** | **516–672 ms = 8.3–9.3×** |
| composed SSJ | 181.5 ms = 12.9× | 879–1246 ms = 15.4–15.8× |

Eigenvalues 4.5–5.9e-15, residuals 2.4–3.1e-15. 165 tests pass. The
campaign's CPU standing has moved 28–42× → 16–18× (SSJ line, attempts 1–15)
→ **6–9× at full accuracy** (purification line, attempts 16–18, from one
steered question). The notebook sync is the last engineering item.

### 19. fp32 splits, and the refinement ladder that makes them exact

The champion's dominant cost is the full-size SP2 projector (~55 gemms, half
the wall). #17 proved mixed purification freezes fp32 subspace error — but
that proof is about the purification loop *in isolation*, and the solver now
ends with an IPT polish that consults A. This tick tested the refutation's
own escape clause.

**First attempt: fp32 SP2 + k polish steps.** The ladder stalled at ~1e-8,
*non-monotonically* (k=1: 3e-9, k=2: 1e-7). Mechanism: the polish's
correction C ≈ antisymmetric is a first-order rotation applied WITHOUT
re-orthonormalization, so V's orthogonality defect lands at O(err²) ≈ 1e-8 —
and the next polish step works in a frame carrying exactly that congruence
error. #15's certificate lesson, resurfacing at a smaller scale: a
non-orthonormal frame poisons the next iteration at its defect².

**The fix follows from the mechanism: alternate.** One Newton–Schulz step
(2 gemms) between polishes takes a 1e-8 defect to 1e-16, restoring exact
error-squaring — measured per step at n=800: **2e-4 → 3e-8 → 2e-12 → 2e-15.**
The general shape is worth stating: *iterative refinement for the
eigenproblem is a consult-A step and a re-orthonormalization, alternating;
skip either and you floor.* (IPT alone escapes this only because its v_jj=1
normalization plays the role of the second step within its basin.)

**Shipped as `purify_eigh(precision="mixed")`** — fp32 SP2 (sgemm rate,
convergence checked every 3rd iteration; the O(n²) check was ~50 needless
passes), loosened split net (fp32 splits legitimately carry ~1e-7), two
polish steps with the NS interleave. Verified post-ship, clean runs
(5.9% / 0.4%):

| | n=400 | n=800 |
|---|---|---|
| `mixed` (new champion) | **84.4 ms = 4.9×** | **461.1 ms = 6.7×** |
| `full` (#18) | 105.2 ms = 6.1× | 591.7 ms = 8.5× |

GOE accuracy 3.7e-15 / 5.7e-14. **The documented boundary:** tight-cluster
residuals floor at ~1e-10 on the mixed route — the polish guard rightly
skips intra-cluster corrections, so fp32-induced mixing inside a cluster
stays. Eigenvalues remain 1e-15 there. `precision="full"` is the default and
keeps 3e-15 residuals everywhere; the test suite pins both routes and the
boundary. 166 tests pass.

Campaign standing on the cold CPU solve: 28–42× (start) → 16–18× (SSJ line)
→ 6–9× (#17–18) → **4.9–6.7×** — inside 5× of LAPACK at n=400, from a
pure-gemm method with a global basin.

### 20. The structural rethink — coarse plus ladder, with a measured boundary

Steered: rethink the algorithm structure in depth. The audit: the champion
spends ~55 gemms per single bit of spectral partition; the deep justification
is that matrix-matrix iteration reaches polynomial degree 3^k in 2k gemms —
exponentially more filter per gemm than any thin/Krylov evaluation, which is
*why* iterating on the matrix beats applying filters to blocks. Against that,
#19's ladder squares error per ~5-gemm pair. Hypothesis: if the ladder's
basin were ~mean-gap/‖A‖ ~ 1/n, the coarse stage could be nearly free and
the ladder would BE the algorithm.

**E1 refuted the hypothesis and produced the boundary.** Corrupting an exact
eigenbasis by an exact rotation of scale ε: the ladder converges from
fp32-quality error and stalls proportionally (≈0.2·ε after six pairs) from
ε = 1e-2, at every n tested — no 1/n scaling. **The ladder buys the last
7–11 digits, never the first four.** The first four digits are the actual
eigenvalue problem; they are also precision-free, which is where substrate
leverage lives.

**E2, substrate ambush #6: ssyevd runs at dsyevd speed on this box**
(17.2 vs 17.3 ms at n=400; 72.6 vs 69.8 at n=800). The natural "fp32 LAPACK
as coarse" instantiation is pointless HERE — and is the obvious headline on
any substrate where low precision is actually cheap (tensor-core GPUs).

**E3, the refiner validated across the battery** (from fp32 bases at ~3e-8):
two pairs land 1.7e-14 (GOE 400), 1.4e-13 (GOE 800), 6.2e-12 (5-fold ties),
1.0e-14 (zero diagonal); the known 1e-10 intra-cluster floor persists. As a
standalone CPU solver, fp32-LAPACK + ladder = 2.2× dsyevd — pointless when
dsyevd is present, decisive when only a low-precision solver is (the GPU
case, the tracking case).

**Shipped: `ssj.refine_eigh(A, w, V, pairs=2)`** — the ladder as a public
API that upgrades ANY ≳1e-4-accurate basis to fp64 at ~5 gemms per squared
digit, with the basin boundary documented and *pinned by a test that
requires the ladder to fail* from 3e-2 corruption (a future "global refiner"
claim must face it). 168 tests pass.

**The settled architecture, stated once:** eigensolving here = a coarse
supplier (SSJ, purification, LAPACK, a tracked basis — interchangeable,
precision-free, must deliver ≲1e-3..1e-4) composed with the refinement
ladder. #19's champion was already its instantiation; this tick named it,
measured its load-bearing boundary, and shipped its reusable half.

### 21. Profile-driven optimization, and the two guards it forced

Component profile of the shipped mixed champion at n=800 (sums 382 of the
461 ms measured): SP2 177, polish×2 92, extraction 68, leaves 26, B-form 11,
G 7.5. The batch, in profile order:

* **Tight bounds.** Gershgorin measured **12.5× loose** on GOE n=800, and
  the SP2 seed slope ∝ 1/enclosure-width, so looseness costs ~log₂(12.5) ≈ 4
  doubling iterations per split. Power iteration on A² (robust to the ±λ
  near-ties of flat spectra), inflated 25%, clipped to Gershgorin.
* **In-place SP2**: preallocated ping-pong buffers; the P ← P² branch is a
  pointer swap; the loop's temporaries had measured ~1 ms/iteration.
* **In-place polish**: the guard `|gap| < 10³|W|` is exactly `|C| > 10⁻³`
  on the computed correction — one mask, which also absorbs the inf (gap→0)
  and NaN (0/0) cases; C reuses the gap buffer.
* **Trailing NS dropped**: the final polish leaves defect (input err)² ≈
  1e-16; the closing NS step was 2 wasted gemms.
* **Cached G** (deterministic anyway; ~8 ms at n=800).

**The batch broke two spectra, and both breaks bought permanent guards:**

1. A power estimate is NOT an enclosure. On a clustered spectrum the
   under-enclosed edge eigenvalue **survived the McWeeny warmup** (its
   attractor basin extends to ~1.37) and exploded only later under SP2's
   squaring branch — fp32 NaN with a finite-looking warmup. The divergence
   guard now lives inside the periodic convergence check (non-finite or
   1e3×-growing error → retry the whole run from the exact Gershgorin
   enclosure, which provably cannot spill).
2. The retuned bounds moved a split point **into a 1e-9 cluster** — the
   fp32 route cannot see structure below ~1e-7 to avoid the cut, and no
   polish can reunite a cluster split across two blocks. Caught by a free
   post-hoc **boundary audit**: after recursion the boundary gap is known;
   below 1e-7·scale → `eigh` fallback. (The suite caught both breaks —
   sixth and seventh saves for the spectrum battery.)

**Result, all runs clean (0.4–3.6% contamination), accuracy asserted:**

| n | mixed (was) | now | full | LAPACK |
|---|---|---|---|---|
| 400 | 84.4 ms / 4.9× | **53.2 ms / 4.4×** | 71.2 / 5.9× | 12.1 ms |
| 800 | 461.1 ms / 6.7× | **283.1 ms / 5.3×** | 395.8 / 7.4× | 53.8 ms |
| 1600 | — | **1761 ms / 6.7×** | 2532 / 9.6× | 264 ms |

Residuals 1.3e-15 / 1.7e-15 / 2.0e-13. First n=1600 measurement of the
campaign. 168 tests pass. Cold-CPU arc: 28–42× → 16–18× → 6–9× → 4.9–6.7× →
**4.4–6.7× across a 4× size range.**

### 22. The minimality audit — every remaining step earns its place

Steered: cut unnecessary steps. Four candidates where a step's *existence*
was questionable. Three died with mechanisms, one survived at noise level.

**Cut 1 — run the whole split pipeline in fp32** (Y-fill, extraction QR,
B-form; the ladder digests 1e-7-class error, so fp64 exactness there looks
wasted). Refuted twice over: it measured *slower* (113 vs 53 ms at n=400)
and less accurate. The speed: **sgeqrf runs at dgeqrf speed on this box
(1.00×) — substrate ambush #7**, extending #6's ssyevd finding; fp32 QR
saves nothing and the casts cost. The accuracy: the fp64 QR of
[P·G₁, (I−P)·G₂] is *load-bearing exactness* — it manufactures an exactly
orthogonal split basis out of an inexact projector, hiding P's fp32 defect
from everything downstream. Make it fp32 and the defect leaks through.

**Cut 2 — replace the McWeeny warmup with an exact rank from Sylvester
inertia** (one ssytrf; the warmup's only job is making round(trace)
trustworthy). The inertia count is exact on every spectrum tested — and
useless here: ssytrf costs as much as the 12 sgemms it replaces (32 vs
~30 ms at n=800; more at n=400). Correct idea, wrong substrate.

**Cut 3 — conditional second polish pair.** The free signal available after
pair 1 (orthogonality defect of V) measures the wrong thing: it sits at
(input err)² ≈ 1e-14 even when the *residual* still needs the second pair.
Skipping on it broke three of five battery rows. A residual-aware skip would
need the polish to export its own correction norm — possible, not free, not
worth it against a ~30 ms step.

**Cut 4 — bounds at 7 power iterations instead of 15**: survived (the 25%
inflation and the divergence guard already bracket the estimate), measured
within noise. Kept.

**The law underneath, stated once: on this stack, reduced precision pays
only through gemm.** sgemm is 2× dgemm, but ssyevd (#20), sgeqrf (#22) and
ssytrf (#22) all run at fp64 speed. Every place the campaign's mixed
precision ever won — SSJ's fp32 sweeps, SP2's fp32 projector — is a place
where the flops were pure gemm. Every place it lost is a factorization.
Tensor-core substrates invert all of these verdicts simultaneously, which
is what the notebook exists to measure.

**Verdict: CPU optimization of the champion is converged.** The pipeline —
7-iteration bounds, fp32 SP2 with guards, fp64 randomized extraction, fp64
B-form, LAPACK leaves, two-pair ladder with mid NS, three safety nets — has
no removable step and no substitutable kernel left on this substrate.
53.7 / 300.4 ms (4.2× / 5.3× LAPACK) at n=400/800, full accuracy, clean
runs. 168 tests pass. What remains is the GPU notebook and the proof.

### 23. The compiled implementation — and what it settles

Attempt #13 claimed the algorithm sits ~2–3× LAPACK in *flop* units and the
rest of its wall gap is NumPy substrate. That is a falsifiable prediction and
`csrc/purify_eigh.c` tests it: the same algorithm in C, linking the **same**
OpenBLAS NumPy uses (`scipy_openblas64`, 64-bit int, `scipy_*_64_` symbols),
with dsyevd benchmarked through the same binary — so nothing about the
comparison is cross-library. That mattered: #13's near-miss verdict on
symmetric BLAS came from SciPy's *wrapper* being 2.6× slower than NumPy's
`@`, not from the kernels.

**What C buys, in the profile's order:** `dsymm` for A·X (A is symmetric —
half the flops, and NumPy's `@` cannot express it) in both the B-form and
every polish step; `dsyrk` for VᵀV in the Newton–Schulz step; the polish's
correction built in **one fused pass** over n² where NumPy needs ~6
(subtract, divide, abs, compare, mask-assign, fill-diagonal); zero allocation
in any loop; SP2 ping-ponging two preallocated fp32 buffers.

**Result — quiet box, interleaved min-of-5, contamination 0.2–0.7%, every
configuration accuracy-checked before timing:**

| n | C | Python (#22) | dsyevd |
|---|---|---|---|
| 400 | **57.8 ms = 3.3×** | 4.2× | 17.6 ms |
| 800 | **315.0 ms = 4.3×** | 5.3× | 73.9 ms |
| 1600 | **1880.7 ms = 5.2×** | 6.7× | 361.7 ms |

Accuracy is *better* than the Python route (|Δλ|/‖A‖ 9.0e-16–1.3e-15,
residual 1.5e-15–1.8e-15) — the fused polish avoids intermediate rounding.
The full battery passes: GOE, exact 5-fold ties, zero diagonal, and the 1e-9
cluster at its documented fp32-route floor (2.3e-10), which the harness
carries as a *per-case documented bar* rather than silently tightening.

**The verdict on #13, stated honestly: half right.** ~1.3× of the gap was
implementation and is now recovered. The rest is not — and the shape of the
remainder says why: **the ratio grows with n (3.3 → 4.3 → 5.2)**, so this is not a
constant implementation tax but the algorithm's own asymptotics losing to
dsyevd's improving constant. The purification recursion does O(n³) work per
split with a large constant (~55 sgemms) plus two O(n³) LAPACK leaves;
dsyevd does one O(n³) pass whose efficiency *rises* with n. #13's flop model
under-counted because it priced the split's gemms at gemm rate but ignored
that a *bisection* pays them at every level while dsyevd pays once.

Build: `cd csrc && make run` (auto-detects NumPy's OpenBLAS; override with
`make BLAS=...`). Zero warnings under `-Wall -Wextra`. The C port is also the
natural GPU starting point: every kernel it calls has a cuBLAS twin, and the
one place it spends fp32 (SP2) is exactly where tensor cores pay 8–16×.

### 24. Non-symmetric: what transfers, what breaks, and why the prize is bigger

Asked directly whether the method works on non-symmetric input. It decomposes
into three independent questions, and they have three different answers.

**The splitter does NOT transfer.** Purification's seed maps eigenvalue
lambda to z = 0.5 + c(mu - lambda); for complex lambda that z is complex, and
`z <- 3z^2 - 2z^3` has attracting basins around 0 and 1 that are *bounded*
regions of C with a Julia boundary through 1/2. Measured:

| spectrum | purification | matrix sign |
|---|---|---|
| non-sym, real spectrum, cond(X)=10 | converges, 31 it | converges, 22 it |
| non-sym, real spectrum, cond(X)=1e4 | **diverges** | converges (degraded) |
| real Ginibre, \|Im\|/\|lambda\| = 0.98 | **diverges** | converges, 18 it |

So purification is a real-line instrument. **The matrix sign function is the
correct substitute in the same architectural slot** — gemm-only, globally
convergent, splitting on the line Re(z) = sigma instead of a point on R — and
`ssj.sdc` already implements it with the same Newton -> Newton-Schulz handoff
pattern the rest of the repo uses.

**The split degrades gracefully, and the mechanism is exactly ||P||.** For
symmetric A the spectral projector is orthogonal, so ||P|| = 1 identically and
`Q^T A Q` is block *diagonal*. For non-symmetric A the projector is oblique:
range(P) is invariant, so the similarity is block *triangular*, and the
backward error scales with ||P|| = the eigenvector conditioning:

| cond(X) | sign iters | \|\|P\|\|_2 | \|\|P^2-P\|\| | \|\|A21\|\|/\|\|A\|\| |
|---|---|---|---|---|
| 1e1 | 22 | 3.2e0 | 1.7e-14 | 1.5e-14 |
| 1e3 | 60 | 2.0e2 | 1.4e-08 | 1.9e-13 |
| 1e4 | 60 | 1.8e3 | 4.9e-05 | 4.6e-12 |
| 1e6 | 60 | 1.5e5 | 3.1e-04 | 1.6e-09 |

It never *fails*; it loses digits in proportion to ||P||. **That single number
is the whole difference between the symmetric and general problems here**, and
it is why the symmetric solver's split needs no conditioning caveat at all.

**The ladder half-transfers.** The consult-A IPT step is basis-general and
works: an fp32 `dgeev` presolve at residual 1.0e-8 refines to **2.8e-16 in 4
iterations** (measured rate 5.6e-3). The Newton-Schulz half does not transfer
— non-symmetric eigenvectors are not orthonormal, so re-orthonormalizing
would destroy the answer. The ladder degenerates to plain IPT, which is what
`ssj.refine_eig` already is.

**And the prize is ~7x larger.** Measured on this box:

| n | dgeev | dsyevd | ratio |
|---|---|---|---|
| 200 | 181 gemm-equiv | 25 | **7.3x** |
| 400 | 131 gemm-equiv | 17 | **7.8x** |

The whole campaign's thesis is that a gemm-only method wins where the
incumbent runs far below gemm efficiency. On the symmetric side dsyevd is only
17-25 gemms and the best this repo achieved is 3.3x *slower* than it. On the
non-symmetric side the incumbent is 131-181 gemms — the same structural
argument has seven times more room, and `sdc.py` was already aimed there.

**Two harness bugs caught before any of this was believed**, both mine: I fed
`matrix_sign`'s `(S, iters)` tuple into arithmetic inside a bare `except`,
which silently reported "sign FAILS everywhere"; and I built the split basis
with a pivoted QR of `[P, I-P]`, whose column reordering destroys the range
separation, reporting ||A21|| = 2.6e-01 for the *symmetric* case where it must
be ~1e-15. Both were caught by asking why a number contradicted theory rather
than by the numbers looking bad. Eighth entry in the measurement-lessons
family: a bare `except` around a numerical call converts a type error into a
scientific claim.

### 25. The non-symmetric race — the opening is real, the implementation is not

#24 established the non-symmetric incumbent is ~7x weaker relative to gemm
(dgeev 131–181 gemm-equivalents against dsyevd's 17–25) and that the sign
function is the right splitter there. This tick ran the race.

**The leaf lesson transferred, and is shipped.** `sdc_eigvals` defaulted to
`min_block=2` — recursing to 2x2 blocks, the exact choice #17 identified as
the symmetric solver's biggest single loss. Measured on Ginibre n=400:

| leaf | wall | vs dgeev | eigenvalue error |
|---|---|---|---|
| 2 (old default) | 1610 ms | 21.1× | 1.6e-13 |
| 32 | 1902 ms | 25.0× | — |
| 100 | 1140 ms | 15.0× | — |
| **n/2 (new default)** | **407 ms** | **5.3×** | **5.8e-14** |

**4x faster and no less accurate** (on the planted real-spectrum case, 3.0e-13
→ 4.6e-15). Each level pays a full-size sign iteration to avoid a dense solve
that costs milliseconds — deep recursion buys splits nobody needed. Default
changed to `max(2, n//2)`, pinned by a test; 168 tests pass.

**And then the real finding, which is a contradiction.** Operation counts,
load-immune, for the whole solve at n=400 (leaf=n/2):

| | gemm | inv | qr | total gemm-equiv |
|---|---|---|---|---|
| SDC | 25 | 10 | 1 | **88** |
| dgeev (measured) | — | — | — | **89** |

**Parity in the ledger. 5.3x loss at the wall.** The kernel weights are not
the problem — an inverse measured 5.02 gemm-equivalents against the model's
5.35. The cost is localized instead: timed min-of-3 with warmup in one run,
**a single `matrix_sign` call at n=400 costs 418 ms against dgeev's entire
65.9 ms solve — 6.3x the incumbent for one split** — and sign plus the two
leaves account for 90% of the 496 ms total.

**Where inside `matrix_sign`, I could not establish this tick, and say so
rather than guess.** A denormal-arithmetic hypothesis was specific and
testable and is **refuted** (zero subnormal entries across all 15 iterations;
|X| stays in [1e-7, 1.03]). My per-iteration decomposition attributed ~990
gemm-equivalents to 9 inverses, which contradicts the 5.02 measured in
isolation — but that harness timed each piece single-shot, cache-cold and
un-warmed inside the loop, which is not a measurement, so it is discarded
rather than reported. Ninth measurement lesson, and a new shape: *a
decomposition whose parts contradict an isolated measurement of the same
kernel is measuring the harness.*

**Verdict.** `sdc.py`'s founding claim — that a method doing several times
more arithmetic still wins when the arithmetic is gemms — is **true in the
operation ledger and false at the wall**, and the entire discrepancy sits in
one function. That makes the non-symmetric side the campaign's best-defined
remaining target: not a search, a single profiling job on `matrix_sign` with
the discipline the symmetric side already paid for (#21's component profile,
#23's compiled port). If it recovers even half, SDC reaches parity with dgeev
in wall as it already has in operations.

### 26. Column-splitting IPT — the gate is a MAX, and the fix is exact

**The question (user).** *Can IPT work on the well-separated eigenvalues while
other algorithms handle the remaining dimensions?* Yes — and the interesting
part is that the composition is exact rather than approximate, and that the
screen it needs turns out to be a safety property rather than an optimization.

**The obstruction is a statistic, not a matrix.** IPT's admission test is a
global rate, `rho = max` over ALL pairs of `|W_ij| / |d_i - d_j|`. That MAX is
brittle: one tight cluster of k eigenvalues sends `rho` to infinity and
disqualifies the entire matrix, while n − k columns sit at `rho_j ~ 1e-3` and
would converge in three iterations. Nothing is wrong with those columns.

**Why the restriction is exact.** The IPT map is column-separable — column j
reads A and column j, never any other column — so restricting to a subset S is
not deflation, locking, or projection onto an approximate invariant subspace.
It is the same iteration, unchanged, on fewer columns. The repo already had
the pieces (`ipt_rate_columns`, `ipt_eig_partial`); what was missing was the
composition. Shipped as `ipt_hybrid_eigh`: screen per column, IPT on
`{rho_j < gate}`, deflate |C| random vectors against the converged
eigenvectors, solve the |C|×|C| projected block densely, concatenate.

**Correctness result — decisive.** Well-separated diagonal plus one tight
k-cluster, global `rho` from 0.87 to 130:

```
   n   k  rho_glob  |C|   hybrid dlam   plain ipt_eigh
 400   5   8.3e+00    5        2.3e-15   FAIL 7e-07
 400  20   1.1e+02   20        2.1e-15   FAIL 1e-04
 800   5   8.7e-01    5        1.3e-15   FAIL 2e-07
 800  20   1.3e+02   20        2.3e-15   FAIL 1e-06
1600   5   6.1e+00    5        6.2e-15   FAIL 2e-06
1600  20   1.0e+02   20        8.6e-15   FAIL 1e-05
```

**Speed result — parity, and the reason is structural.** 0.97×–1.08× dsyevd
(contamination 0.2–3.8%, accuracy asserted before every timing). Removing k of
n columns from IPT saves k/n of IPT's cost, so the composition inherits IPT's
economics and cannot improve them. **The column split buys ADMISSION, not
throughput** — it converts a hard global gate into a soft per-column one, and
the throughput still comes from `rho_j` on the columns that pass.

**The finding that matters most, and it is a hazard.** My first instinct was
to delete the screen: `rho_j` is a one-hop optimistic heuristic predicting
something `ipt_eig_partial` measures exactly and reports in `info['failed']`,
so why not run IPT on every column and let it flag its own casualties? Measured
at n=1600 with a 20-cluster, the unscreened path is 2–3× slower (divergent
columns burn max_iter) **and wrong**: eigenvalue error 3.7e-07, orthogonality
defect 2.35. Chased it down rather than backing off, with a stated prediction
and a stated refutation condition. Confirmed exactly: **1582 columns flagged
converged, rank 1581 — columns 800 and 801 returned the identical eigenvector
to |⟨v_j,v_p⟩| = 1.000000 and the identical eigenvalue to 12 digits.**

That is not a bug in the flag. Both vectors genuinely ARE fixed points of their
own column's map, so both step norms genuinely are tiny. **Column separability
— the property that makes the partial solve exact — is the same property that
removes any mechanism preventing two columns from landing on the same
eigenpair.** Per-column convergence is a statement about one column's residual
and can say nothing about the basis being complete. Both offenders had
`rho_j` = 1.30 and 3.67 and are rejected by the screen. Pinned as a test that
asserts the collapse still happens, so the screen's rationale cannot quietly
rot.

**A shipped speedup fell out of the profiling.** The component breakdown found
the overhead was not where I assumed — `ipt_eig_partial` costs only 1.04–1.07×
the full path, and deflation + QR + projection total under 7 ms at n=1600 —
but `ipt_rate_columns` was 22–47% of the run. It was a Python loop over
columns building seven O(n) temporaries and forcing a scalar sync each time;
its cost was never its flops. Blocked vectorization (blk = 64, chosen over a
grid; peak memory stays O(n·blk)), **bit-identical to the loop on every case
tested, 2.4×–4.6× faster**, now the shipped NumPy path. The GPU keeps the loop
— untested kernels do not ship here (#7).

**Tenth measurement lesson, and this one cost a real bug.** The vectorized
form leans on IEEE (`w/gap` already yields `+inf` for the divergent case)
instead of branching, and `np.nan_to_num`'s DEFAULT collapses `+inf` to
1.8e308. That would silently turn an exactly degenerate coupled pair — the
divergent case this screen exists to catch — into a large finite rate that
passes any gate. My four bit-identity checks all passed, because no test
matrix had an exact degeneracy. **A bit-identity check certifies only the
inputs it was given; the case a function exists to catch is the one to feed it
first.** Caught by writing the degeneracy test, not by the benchmark.

179 tests pass.

### 27. SDC in C — the ledger was right and NumPy was the whole gap

**The contradiction this settles.** #25 left the non-symmetric side with a
clean paradox: SDC-by-sign has *operation-count parity* with dgeev (88
gemm-equivalents against 89) yet lost 7.7–14× at the wall, with the entire
discrepancy inside one function, `matrix_sign`. Either that gap was NumPy
substrate — as the symmetric side's partly was (#13/#23) — or the operation
model was wrong. A compiled port linking the SAME OpenBLAS decides it.

**Result: the model was right.** `csrc/sdc_eig.c` + `csrc/sdc_bench.c`,
Ginibre, interleaved min-of-5, contamination 0.0–1.3%, accuracy asserted
first:

```
   n   sdc_c      dgeev    ratio    dlam      Python was
 200   30.1 ms    17.0 ms  0.56x    3.8e-14   0.07x
 400  112.9 ms    68.8 ms  0.61x    4.0e-14   0.13x
 800  559.6 ms   341.4 ms  0.61x    6.1e-13   —
```

**From 7.7–14× slower to 1.6–1.8× slower — a 5–8× implementation gain**,
against the symmetric port's 1.3× (#23). The non-symmetric side was far more
NumPy-bound than the symmetric one, and now nearly all of it is recovered.

What did it, in the order the flop model ranks it: `dgetri` for the inverse
instead of `lu_solve` against a materialized n×n identity (4n³/3 against
2n³/3 + 2n³, and no identity ever formed); `‖X²−I‖_F` computed in one pass
over X² without materializing the difference, on *every* iteration; the
Newton combination `(μX + X⁻¹/μ)/2` as one fused pass instead of four; the
Newton–Schulz target `1.5I − 0.5X²` built in place over X²; and no allocation
anywhere in the iteration, which ping-pongs two preallocated buffers.

**The phase attribution, now trustworthy.** Accumulated wall across whole
timed runs rather than #25's single-shot cache-cold decomposition, and stable
across three sizes: **sign 66–71%**, leaves 15–19%, pivoted QR 13–14%, the
two `QᵀAQ` gemms 2.6%. One *isolated* sign evaluation costs 0.79–1.04× the
entire dgeev solve — down from 6.3× in Python, which is exactly the term that
moved.

**And the remaining gap has a name: 2 sign calls per solve where 1 would
do.** Since sign is ~2/3 of the run, the second call is ~1/3 of the total.
I guessed in this entry's first version that the centred shift `tr(A)/n` was
being *rejected* and a perturbed shift retried. **That guess was wrong, and
#28 measured it**: across 12 Ginibre configurations not one shift is ever
rejected — zero rank failures, zero singular iterates, zero backward-error
rejections, zero wasted iterations. The second call is a second *split*, and
the cause is the leaf. See #28.

**A test-matrix trap, caught by making the case validate itself.** My first
`planted_real` used strictly-upper entries of magnitude 0.3, and SDC "failed"
on it at 2.7e-03 to 6.9e-02 with all 12 shifts rejected. The decisive check
was not to debug SDC but to ask what dgeev does: **dgeev itself misses the
planted spectrum by 1.5e-01 to 2.0e-01 there.** The construction is
hyper-non-normal and its eigenvalues are simply ill-conditioned, so nothing
measured on it says anything about a solver. At 0.005 both dgeev and SDC
recover the planted values to 1.7e-14 (n=200) to 8.5e-13 (n=800), and SDC
needs a single sign call. The bench now prints the dgeev-vs-planted distance
every run so the case cannot silently rot back into meaninglessness.
**Eleventh measurement lesson: when a solver fails a synthetic case, measure
the INCUMBENT on that case before believing the failure is the solver's.**

Also carried over from the symmetric side and load-bearing here: spectra are
compared by nearest-match, never by sorting. A real matrix has exact
conjugate pairs whose real parts tie, so a lexicographic sort reports errors
of order 2|Im λ| that do not exist — that bug cost a full re-measurement
earlier in this session and is now commented at the comparison site.

### 28. The leaf was off by three — and my #27 diagnosis was a guess

**Two findings, and the first one is about method.** #27 closed by naming the
remaining gap: 2 sign calls per solve where 1 would do, worth ~1/3 of the run.
It also *explained* that gap — the centred shift `tr(A)/n` is rejected and a
perturbed shift retried — and the explanation was a guess, reasoned from the
fact that Ginibre's spectrum is a disk about the origin so a split at
Re(z) = 0 puts eigenvalues near the splitting line. It reads like a finding.
It was wrong.

Counters on every guard (degenerate rank, singular iterate, backward-error
rejection, iteration limit) plus the iterations burned inside rejected
attempts, across 12 Ginibre configurations at n = 200/400/800:

```
   n   seed  sign calls  rank  singular  resid  maxiter  wasted iters
 200    1-4      2         0       0       0       0          0
 400    1-4      2         0       0       0       0          0
 800    1-4      2         0       0       0       0          0
```

**Not one shift is ever rejected.** The second sign call is a second *split*.

**The actual cause: the leaf.** The default was `max(2, n//2)`, chosen in #25
to mean "one split, both halves to dgeev". It does not mean that. The split
returns `r = trace(P)`, which lands NEAR n/2 but essentially never ON it, so
one half comes back a few rows too big, fails the leaf test, and buys a whole
second full-size sign iteration to shave a dgeev that was already cheap.

**The fix and its measurement.** Leaf sweep on Ginibre, accuracy asserted
before timing — 2 sign calls at 0.50n, 1 at every fraction from 0.55n to
0.90n:

```
   n    leaf 0.50n      leaf 0.60n     dgeev     ratio
 200      28.9 ms         27.5 ms     15.2 ms   0.53x -> 0.55x
 400     110.4 ms         94.8 ms     64.0 ms   0.58x -> 0.68x
 800     581.0 ms        522.8 ms    339.6 ms   0.58x -> 0.65x
```

Confirmed end to end at the new default (3n/5), 1 sign call at every size:
**0.60×/0.67×/0.65× dgeev**, dlam 3.4e-14 / 3.9e-14 / 6.1e-13, contamination
0.4–3.0%. The Python solver had the identical defect and gains more, being
iteration-bound rather than kernel-bound: **412 → 207 ms at n=200 and
1315 → 748 ms at n=400.** Both defaults are now 3n/5.

The gain is FLAT from 0.55n to 0.90n, so the constant is not delicate; what
matters is being strictly above n/2 while staying well below n, so a genuinely
LOPSIDED split still leaves a big block that recurses. **Third appearance of
the leaf lesson (#17, #25)** — and the first two were both "the leaf is too
deep", which is exactly why this one was invisible: it is the same mistake
wearing the opposite sign.

**A cross-check that now closes.** #27 reported the isolated `matrix_sign`
call and the in-solve attribution disagreeing (49.2 vs 74.8 ms at n=400) and
declined to interpret it. With one sign call per solve they agree exactly —
15.7 vs 15.7, 49.2 vs 50.8, 321.6 vs 314.2 — so the discrepancy was entirely
the second call, not a harness artifact. Declining to explain it at the time
was the right call.

**Twelfth measurement lesson, and it is about writing, not measuring.** The
counters cost twenty minutes; the wrong explanation would have sent the next
reader hunting a retry that does not exist. *An explanation adjacent to a
measurement inherits none of its credibility.* #27's number (2 sign calls)
was solid and its cause was invented, and nothing in the prose marked the
seam. The entry has been corrected in place rather than left standing with a
footnote.

**Where sign now sits, and the next target.** Phase split at the new default:
sign 54–61%, leaves 22–30%, pivoted QR 11–13%, the `QᵀAQ` gemms 2.4%. One
sign evaluation costs 0.77–1.02× the *entire* dgeev solve, so sign must get
cheaper for SDC to reach parity. The specific opening, from the flop model:
a Newton step (gemm + `dgetrf` + `dgetri` ≈ 4n³) and a Newton–Schulz step
(two gemms ≈ 4n³) cost the SAME arithmetic, but NS is pure gemm while Newton
is two factorizations that do not run at gemm rate — so at equal flops NS
should be wall-cheaper, and the switch threshold `ns_switch = 0.6` is
probably conservative. At n=800 the run spends 10 Newton steps against 6 NS.
Sweeping that threshold is the next tick and it is a one-parameter job.
