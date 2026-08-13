# Measured results — Python/NumPy implementation

Environment: Python 3.11, NumPy 2.4 / OpenBLAS, 4 threads on a shared 4-core
container (wall times carry that machine's run-to-run noise; sweep counts and
accuracies are stable). Every run starts cold from $X_0 = I$ with tolerance
$10^{-13}$ on $\|\mathrm{offdiag}(X^\top A X)\|_F / \|A\|_2$, where $\|A\|_2$
is a power-iteration estimate (see "Refinements" below). Battery and scaling
rows reproduce with `python3 validate.py --full`; mechanism experiments with
`python3 experiments.py`. Error columns as in [RESULTS.md](RESULTS.md):
$d\lambda$ = max eigenvalue error vs LAPACK, resid =
$\|AV - V\Lambda\|_F/\|A\|_2$, ortho = $\|V^\top V - I\|_F$.

The reference measurements from the original Julia implementation are in
[RESULTS.md](RESULTS.md); rows here use the same matrix families (independent
random constructions, so seeds differ).

## Convergence battery

| input | sweeps (ref: Julia) | $d\lambda$ | resid | ortho |
|---|---|---|---|---|
| diagonal + coupling at 1× the level spacing, $N=200$ | 10 (10) | 6.2e-15 | 5.6e-15 | 2.4e-15 |
| same, 5× | 15 (20) | 1.1e-14 | 6.9e-15 | 3.9e-15 |
| same, 100× | 20 (21) | 3.6e-14 | 7.9e-15 | 4.6e-15 |
| GOE, $N=200$ | 20 (21) | 5.1e-15 | 7.4e-15 | 4.7e-15 |
| GOE, `method="gemm"` (factorization-free) | 20 (21) | 5.2e-14 | 1.0e-13 | 2.3e-13 |
| zero diagonal — every gap $=0$ at sweep 1 | 21 (21) | 9.3e-15 | 7.2e-15 | 4.8e-15 |
| tridiagonal Toeplitz $(2,1)$, $N=200$ | 24 (37) | 4.4e-15 | 6.2e-15 | 3.7e-15 |
| Wilkinson $W_{21}^+$ | 14 (16) | 5.3e-15 | 1.5e-14 | 7.0e-16 |
| graded spectrum $2^{-i}$, $N=200$ | 48 (20) | 2.5e-16 | 9.3e-14 | 1.4e-14 |
| ten exact 5-fold degeneracies, $N=500$ | 74 (74) | 3.6e-15 | 7.1e-14 | 2.8e-14 |

New rows — the same map extends to complex Hermitian input with
$K_{ij} = \frac12\arctan(2|B_{ij}|/(d_j-d_i))\cdot B_{ij}/|B_{ij}|$:

| input | sweeps | $d\lambda$ | resid | ortho |
|---|---|---|---|---|
| GUE, $N=200$ | 22 | 6.7e-15 | 4.9e-15 | 4.8e-15 |
| GUE, $N=200$, `gemm` | 22 | 3.7e-14 | 2.6e-14 | 2.2e-13 |

Sweep counts land within a few of the Julia reference throughout (different
random constructions), including exact matches on the two structurally forced
cases: zero diagonal (21) and the ten 5-fold degeneracies at $N=500$ (74).
The two larger gaps: our Toeplitz (2,1) run converges in 24 sweeps vs 37
(construction is deterministic, so this difference is real and unexplained —
possibly a different $N$ in the reference); our graded run takes 48 vs 20
(the reference's construction of "spectrum $2^{-i}$" may grade the matrix
entries rather than conjugate by a random basis). Both converge to full
accuracy.

## Convergence trajectory

GOE, $N=200$, `method="auto"`: $\mathrm{off}(B)/\|A\|$ per sweep:

```
7  6.8  6.5  5.9  5.3  4.6  3.9  3.2  2.6  2  1.5  1
0.63  0.34  0.14  0.035  0.0027  2.3e-05  3.4e-07  3.4e-12  6.1e-15
```

Same shape as the reference: a linear global phase, then a quadratic tail
where each sweep roughly squares the error.

## Sweep count scales like $O(\log N)$

GOE, cold, `method="auto"`:

| $N$ | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|
| sweeps | 17 | 20 | 24 | 29 | 35 |
| (Julia ref) | 17 | 21 | 25 | 29 | 36 |

≈ +4.5 sweeps per doubling.

## Monotonicity

Across 20 GOE seeds ($N=100$), $\mathrm{off}(B)$ never increased — worst
single-sweep change $-1.6\times10^{-13}$, matching the reference. Our
Toeplitz $(2,1)$ run also shows no excursion (reference reports a
$+2.3\times10^{-3}$ excursion on its Toeplitz run; with our 24- vs 37-sweep
discrepancy on this family, the two runs are evidently not the same matrix).

## Negative results (all measured on GOE $N=200$, reference converges in 21)

| variant | outcome |
|---|---|
| second-order retraction $X(I+K+K^2/2)$ | stuck at off $=7.2$ after 60 sweeps |
| deferred orthonormalization (QR every 2nd sweep) | grows to off $=78$ |
| arctan-compensated step ($K$ scaled by $\sigma/\arctan\sigma$) | stuck at off $=7.2$ |

The first two confirm rows of the reference's negative-results table. The
third is our own candidate "improvement," included as an adversarial test of
the mechanism claim: since the reprojection rotates each $K$-plane by
$\arctan\sigma_\ell$ instead of $\sigma_\ell$, one might pre-compensate by
scaling $K$ so the top plane rotates by exactly its intended angle. It fails
exactly as the mechanism predicts — every other plane over-rotates (by up to
$\sigma/\arctan\sigma \approx 9$ on the first GOE sweep), which is precisely
the aggressiveness the saturation exists to remove. The reference's corollary
survives an attack it hadn't tested: optimize the orthonormalization's cost,
never the step's aggressiveness.

## Refinements found during reimplementation

**1. The retraction must be applied to the product $X(I+K)$** — never as
$X\cdot\mathrm{orth}(I+K)$, which is equivalent for exact retractions and
therefore an easy misreading. With the gemm variant's truncated Newton–Schulz,
only the product form re-measures $X$'s accumulated orthogonality defect
inside $Y^\top Y$ and corrects it each sweep. Measured with the factor form:
apparent $\mathrm{off}(B)$ converges to $10^{-13}$ while the true
orthogonality error of $X$ plateaus at $0.34$ — a silent wrong answer. This
also explains why the spectral cap is 1 and not something larger:
$\sigma(X(I+K)) \le \sqrt{1+\mathrm{cap}^2}\cdot(1+\|E\|)$ must stay inside
Newton–Schulz's $\sqrt{3}$ region *including* the slack $\|E\|$ that the
$0.05\cdot\mathrm{off}$ tolerance permits early on; cap 1 leaves exactly that
headroom ($\sqrt{2}\cdot(1+\|E\|) < \sqrt{3}$ for $\|E\| < 0.22$), while cap 2
starts at $\sqrt{5} > \sqrt{3}$ and diverges, as the reference measured.

**2. Adaptive-depth Newton–Schulz endgame, targeted at the quadratic-tail
scale.** The spec's endgame (a single fixed NS step once $\|K\|_F < \frac12$)
leaves an $O(\|K\|^4)$ orthogonality defect that the last sweep never
corrects. On graded spectra the final $K$ is *not* small at convergence — tiny
bottom-of-spectrum gaps keep angles at $\sim10^{-3}$ while
$\mathrm{off}(B)/\|A\|$ is already at $10^{-13}$ — and the defect surfaces in
the output: measured ortho $= 8\times10^{-9}$ on the graded battery row.
Iterating NS until $\|Y^\top Y - I\|_F < \mathrm{off}^2$ (defect below the
square of the current error cannot disturb the error-squaring tail; an
intermediate attempt with the looser gemm-rule target $0.05\cdot\mathrm{off}$
fixed ortho but stalled the tail for a sweep and broke monotonicity at the
$10^{-7}$ level) restores ortho $= 1.4\times10^{-15}$ at $\sim$2 extra gemms
per endgame sweep. With it, `auto` beats pure QR on wall time *and*
orthogonality (below).

**3. Exact-tie orientation.** At an exact zero gap the vectorized formula
$\frac12\arctan(2B_{ij}/(d_j-d_i))$ evaluates both $(i,j)$ and $(j,i)$ with a
$+0$ gap and returns $+\pi/4$ on both triangles, silently breaking the
antisymmetry of $K$; and a rewrite as
$\mathrm{sign}(g)\cdot2|B|/|g|$ turns the saturation into
$\mathrm{sign}(0)\cdot\infty = \mathrm{NaN} \to 0$, deleting the rotation
exactly where it must saturate (the all-ties $2\times2$ case then never moves
at all). The tie case needs its orientation set explicitly by the triangle.
Both traps were hit and are now covered by tests.

**4. Power-iteration norm estimate.** The stopping rule needs $\|A\|_2$; an
exact 2-norm is an $O(N^3)$ SVD — several sweeps' worth of hidden cost. A
power-iteration estimate can only come in low, which only tightens the
effective tolerance.

## Retraction wall time (GOE $N=1000$, tol $10^{-13}$)

| method | sweeps | time | ortho |
|---|---|---|---|
| `qr` | 31 | 3.47 s | 4.5e-14 |
| `auto` (QR + adaptive NS endgame) | 31 | 3.40 s | 1.7e-14 |
| `cholqr2` | 31 | 10.3 s | 1.7e-14 |
| `gemm` | 32 | 4.20 s | 3.4e-14 | 

The NS endgame gives only $\sim$1.1× here (reference: 1.4×) because this
BLAS build's Householder QR is already fast relative to its gemm.
**CholeskyQR2 is a measured negative on this stack**: its triangular
kernels (potrf/trtri) run far below gemm speed here, so "two gemms plus small
triangular ops" loses to `dgeqrf`+`dorgqr` by 3×. It is kept as an option
because the trade reverses on hardware where gemm dominates factorizations
(the reference's niche (a)); measure before choosing it.

The gemm variant runs 206 raw gemms over 20 sweeps on the battery's GOE row
(reference: 248 over 21) — sweep parity with QR at $\sim$2× the flops, every
flop a gemm.

## Warm starts (eigenpair tracking)

`ssj_eigh(A2, X0=V)` accepts an orthonormal warm start — e.g. the eigenbasis
of a nearby matrix, orthonormalized on entry. GOE $N=1000$, perturbed by a
unit-spectral-norm symmetric matrix scaled by $\epsilon$, warm-started from
the unperturbed eigenbasis (tol $10^{-13}$):

| $\epsilon$ | sweeps | wall | `gemm`-mode raw gemms | LAPACK `eigh` cold |
|---|---|---|---|---|
| 1e-2 | 5 | 0.74 s | — | 0.07 s |
| 1e-4 | 3 | 0.39 s | 20 | 0.07 s |
| 1e-8 | 1 | 0.22 s | — | 0.08 s |

A warm start lands directly in the quadratic tail, so tracking costs
~10–25 raw gemms per update. On this CPU that still loses to simply
re-running `dsyevd` (see below — here a full eigensolve costs only a handful
of gemm-equivalents). The warm-start case is where the method's actual
competitive claim lives: on hardware where an eigensolve costs 50–200
gemm-equivalents, ~20 gemms per tracking update is a 3–10× win, and even the
cold-start gemm variant (~330 gemms at $N=1000$) reaches the boundary of that
range. Unmeasured here — this container has no such accelerator.

## Accelerations explored

Five candidates, implemented and measured. Two won and are in core, one is a
core option for a specific matrix class, two are quantified dead ends.

**Mixed precision (`precision="mixed"`) — in core, ~1.3–1.4× on CPU.** The
linear phase runs in float32 down to $\mathrm{off}(B)/\|A\| < 10^{-4}$, then
the float64 phase warm-starts from the float32 basis (2–5 sweeps to
$10^{-13}$). Sound because the map is memoryless: every sweep re-derives its
angles from a fresh $B$, so low-precision sweeps can only degrade the warm
start, never the final accuracy — all runs finish at $d\lambda \sim 10^{-14}$.
Measured (GOE $N=1000$, best of 3): `auto` 6.15 s → `auto` mixed 4.92 s;
`gemm` 5.98 s → `gemm` mixed **4.60 s** (vs `qr` 6.41 s). The CPU split is
limited by this stack's kernels: sgemm runs 2.5× dgemm but **sgeqrf is not
faster than dgeqrf** (0.93×), which is why the factorization-free `gemm`
method profits most from the float32 phase. On tensor-core GPUs the same
split is worth far more (float32/TF32 throughput 8–16× FP64): in mixed mode
the $N=1000$ solve needs only ~24 FP64 gemms plus ~254 float32 gemms — at
tensor-core rates ≈ 50–60 FP64-gemm-equivalents total, which moves the
*cold-start* case into the competitive range of the GPU section below.

**Power-iteration warm start — in core.** The `gemm` method's spectral-norm
estimate was re-run cold (30 iterations) every sweep — ~1900 matvecs ≈ 2 s of
hidden cost per $N=1000$ solve. The dominant vector now carries across sweeps
(the generator changes slowly), cutting warm estimates to 8 iterations. A
modest underestimate only weakens the cap slightly, inside the
$\sqrt{3}/\sqrt{2}$ Newton–Schulz headroom. This alone made `gemm` faster
than `qr` on CPU.

**Newton–Schulz pre-scaling — in core.** $\sigma(I+K)$ spans
$[1, \sqrt{1+\sigma_c^2}]$ exactly, and the polar factor is scale-invariant,
so dividing $Y$ by $(1+\sigma_c^2)^{1/4}$ centers the singular values around
1 before Newton–Schulz. Battery GOE row: 223 → 185 raw gemms (−17%).

**Unshifted-QR prologue (`prologue=k`) — core option, for graded spectra.**
$k$ steps of $B \leftarrow RQ$ (accumulating $X$) before the first sweep;
off-diagonals decay like $|\lambda_i/\lambda_j|^k$, each step costs about one
sweep. Measured on spectrum $2^{-i}$, $N=200$: **45 → 5 sweeps with
`prologue=3`** (and → 3 with $k=10$), identical accuracy. Useless for flat
spectra (GOE: 20 → 20), harmless. Not a default because it reintroduces a
knob; the default stays parameter-free.

**Over-relaxation — dead end, quantified.** $X \leftarrow
\mathrm{orth}(X(I+\gamma K))$: $\gamma = 1.1$ costs +45% sweeps (20 → 29),
$\gamma=1.5$ costs 3.4×, $\gamma = 2$ diverges. The saturated angles are not
conservative — they are already the right step, and *any* uniform
over-rotation hurts immediately.

**Momentum in generator space — dead end, quantified.** Even with the
saturations intact and the previous generator parallel-transported to the
current frame ($P \leftarrow Q^\top K_{\mathrm{eff}} Q$), every $\beta$
tested slows convergence: $\beta = 0.2/0.3/0.5$ gives 52/67/125 sweeps vs 20.
Together with the reference's Anderson result, this closes the extrapolation
family: it is not the *placement* of the extrapolation that was wrong, the
map simply tolerates no memory. (Consistent with the mixed-precision result —
memorylessness is exactly what makes the float32 phase safe.)

Reproduce with `python3 experiments_accel.py` (sweep counts, deterministic)
and the timing snippets in the tables above.

## GPU (arithmetic and a runnable benchmark, no measurement)

The implementation is backend-agnostic: pass a CuPy array and every operation
— angles, retractions, power iterations — runs on the device.
`bench_gpu.py` measures, per size, the FP64 gemm time, cuSOLVER `syevd`
(the incumbent, in gemm-equivalents), SSJ cold (`gemm` and `cholqr2`
methods), and the warm-start tracking case against a cuSOLVER re-solve.

What the measured gemm counts predict, against the 50–200 gemm-equivalent
range RESULTS.md cites for accelerator eigensolves (dense symmetric
eigensolvers on GPUs are factorization- and memory-bound; the published
cuSOLVER/MAGMA ratios at $N\gtrsim8$k sit broadly in that range, growing
with $N$):

- **Cold start: likely still loses in pure FP64, but `precision="mixed"`
  changes the arithmetic.** ~10 raw gemms per sweep × sweeps growing like
  $\log N$ gives ≈ 450–500 FP64 gemms at $N=10$k — above the range unless
  `syevd` on the target GPU lands at the very top of it. In mixed mode,
  however, ~90% of those gemms run in float32/TF32 at 8–16× FP64
  tensor-core throughput: the measured split (~25 FP64 + ~255 float32 gemms
  at $N=1000$) prices out at ≈ 50–60 FP64-gemm-equivalents — inside the
  range. Mixed-precision cold start on tensor cores is the one configuration
  where SSJ could beat cuSOLVER from cold; bench_gpu.py measures it.
- **Warm-start tracking: the credible win.** 1–5 sweeps ≈ 10–30 raw gemms
  per update is 2–10× under the range, FP64 end to end, with no
  factorization in the loop.
- **CholeskyQR2 should reverse its CPU verdict** on GPU (its triangular
  kernels are small next to its gemms there), making it the retraction to
  try first for the cold-start case.
- **Mixed precision is implemented** (`precision="mixed"`, see
  "Accelerations explored") and validated on CPU at full final accuracy;
  bench_gpu.py includes the mixed cold-start row. FP16 and TF32 variants of
  the low phase remain untested.

These are predictions from measured gemm counts plus published GPU ratios,
not measurements; `bench_gpu.py` exists to replace them with numbers.

## Honest wall-clock context

LAPACK `eigh` (dsyevd) solves the GOE $N=1000$ problem in 0.07 s on this box
against `auto`'s 3.4 s — a $\sim$50× gap from a cold start (the reference
measured $\sim$21× in Julia; ours is wider, consistent with this build's very
fast dsyevd). The niches are unchanged from RESULTS.md: hardware where an
eigensolve costs 50–200 gemm-equivalents, and robustness — a three-line,
parameter-free method with an empirically global basin, native degeneracy
handling, and (new here) identical behavior on complex Hermitian input.

## Not established

Everything in RESULTS.md's "Not established" section still stands: no
convergence proof, novelty vs the parallel-Jacobi literature unverified,
symmetric/Hermitian only. Added by this work: the Toeplitz and graded sweep
counts differ from the reference in both directions for reasons not yet
pinned down (construction details are the prime suspect), and all wall times
are single-machine with the container's scheduling noise.
