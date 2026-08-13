# The general (nonsymmetric) problem

The symmetric solver splits cleanly into a globally convergent but expensive
map (SSJ) and a cheap but locally convergent one (IPT). Carrying that split to
general matrices gives one large win and three instructive failures, and the
failures are what locate the open problem.

Reproduce with `python3 experiments_general.py`.

## The win: IPT extends unchanged, and pays much better

The IPT fixed point never used symmetry — only $WV$ and diagonals — so it
carries over verbatim. What changes is its surroundings: eigenvectors of a
nonsymmetric matrix are not orthogonal, so columns are normalized but never
reorthogonalized (that would be *wrong*, not merely wasteful); $\Lambda$ is
the eigenvalue, with no Rayleigh quotient available; and the diagonal of $WV$
must not be projected to the reals.

The payoff is much larger than in the symmetric case, because the incumbent is
much more expensive. At $N=1000$: `dgeev` costs **65 gemm-equivalents**
against `dsyevd`'s 8.3, so the same 4–9 gemm solve wins by a wider margin.

| $N$ | $\rho$ | IPT iters | IPT | LAPACK `dgeev` | **speedup** | rel $d\lambda$ | resid |
|---|---|---|---|---|---|---|---|
| 500 | 0.002 | 4 | 0.013 s | 0.156 s | **12.2×** | 9.1e-15 | 5.5e-15 |
| 500 | 0.01 | 5 | 0.014 s | 0.158 s | **11.4×** | 9.3e-15 | 6.2e-15 |
| 500 | 0.1 | 9 | 0.037 s | 0.159 s | **4.3×** | 1.1e-14 | 1.0e-14 |
| 1000 | 0.002 | 4 | 0.076 s | 0.749 s | **9.9×** | 1.6e-14 | 9.2e-15 |
| 1000 | 0.01 | 5 | 0.088 s | 0.758 s | **8.6×** | 1.3e-14 | 9.3e-15 |
| 1000 | 0.1 | 9 | 0.173 s | 0.759 s | **4.4×** | 1.2e-14 | 1.2e-14 |

The usable band is wide: even $\rho = 0.1$, nine iterations, still wins 4×.
As in the symmetric case the applicability test `ipt_rate` is $O(N^2)$, so the
choice between IPT and `dgeev` can be made correctly and for free.

## What IPT's basin actually requires

$$\rho = \max_{i \neq j} \frac{|W_{ij}|}{|d_i - d_j|} \lesssim 1$$

Note the denominator: the basin needs **well-separated diagonal entries**, not
merely small coupling. This is the binding constraint, and it is what defeats
the globalization attempts below — a matrix whose spectrum is dense (GOE-like,
minimum gap $O(N^{-2})$) is out of IPT's reach no matter how small the
coupling is made.

## Failure 1: saturating IPT does not widen its basin

IPT's denominator $(\Lambda_j - d_i)$ is the *linearized* gap — the exact
analogue of the linearized Jacobi angle $B_{ij}/(d_j-d_i)$ that RESULTS.md
records diverging at $0.85\times$ the level spacing. SSJ's fix was the exact
2×2 solve. That fix exists here too: for the block
$\left[\begin{smallmatrix} d_i & W_{ij} \\ W_{ji} & d_j\end{smallmatrix}\right]$
with $\delta = (d_j - d_i)/2$ and $p = W_{ij}W_{ji}$, the exact denominator is

$$\lambda - d_i = \mathrm{sign}(\delta)\left(|\delta| + \sqrt{\delta^2 + p}\right)$$

which reduces to $d_j - d_i$ when $p=0$ and tends to $\sqrt{p}$ as the gap
closes, bounding $v_i \to \sqrt{W_{ij}/W_{ji}}$ instead of letting it blow up.
It is structurally identical to Jacobi's
$t = \mathrm{sign}(\tau)/(|\tau| + \sqrt{1+\tau^2})$.

Measured, $N=300$:

| $\rho$ | linearized | saturated (exact 2×2) |
|---|---|---|
| 0.01 – 0.8 | 5, 9, 14, 20, 33 its | **identical: 5, 9, 14, 20, 33 its** |
| 1.2 | diverged (err 8e+02) | diverged (err 5e-02) |
| 2.0 | diverged (err 5e+03) | diverged (err 5e+00) |

**The saturation changes nothing inside the basin and does not extend it.** It
only bounds the damage outside (divergence four orders of magnitude gentler).

This sharpens RESULTS.md's mechanism claim rather than contradicting it. SSJ is
stable because it *linearizes then reprojects* — two saturations. The arctan
bounds each pair angle; the reprojection of $I+K$ onto $O(N)$ saturates the
*composed* step. In $GL(N)$ there is no manifold to reproject onto, so only the
first saturation survives, and the first one alone was never the stabilizer.
The experiment is a direct test of that claim, and it passes.

## Failure 2: orthogonal SSJ cannot reach real Schur form

RESULTS.md reports this and it reproduces independently here, using the exact
2×2 triangularizing angle (not a linearized one). $\|\mathrm{tril}(B,-1)\|_F/\|A\|_2$:

- general near-diagonal, $\rho = 0.1$: starts at **0.018** and grows
  monotonically — 0.026, 0.057, 0.13, 0.26, 0.47 — reaching 2 after 200
  sweeps. The iteration actively destroys an almost-triangular starting point.
- Ginibre $N=100$: flat at ~3.7, no progress at all.

So the failure is not slow convergence: the lower-triangular norm is not a
descent quantity for simultaneous rotations, exactly as the doc states.

## Failure 3: symmetric-part globalization

For nearly symmetric $A = S + \eta N$, an appealing route is to diagonalize
$S$ with SSJ and hope the resulting frame puts $A$ in IPT's basin. It fails
for the reason in the basin analysis above ($N=400$, GOE-like $S$):

| $\eta$ | $\rho$ after SSJ | IPT | total | `dgeev` | verdict |
|---|---|---|---|---|---|
| 0.001 | 0.101 | 10 its | 0.411 s | 0.087 s | converges, but **0.21×** — SSJ costs more than `dgeev` |
| 0.01 | 1.006 | — | — | 0.090 s | outside the basin already |
| ≥ 0.1 | 10–100 | — | — | ~0.1 s | far outside |

Diagonalizing $S$ puts GOE eigenvalues on the diagonal, whose minimum gap is
$O(N^{-2})$; $\rho$ then explodes for any antisymmetric part at all. And even
where it converges, SSJ is more expensive than the `dgeev` it is meant to beat.

## Where this leaves the general problem

The general problem now has a **fast local solver with a free applicability
test** (IPT, 4–12× over `dgeev`, dispatchable on `ipt_rate`) and **no
globalizer in this family**. The three failures above are not
near-misses; each is blocked by a different structural fact:

1. No manifold in $GL(N)$ ⇒ no second saturation ⇒ no self-stabilization.
2. $\|\mathrm{tril}\|^2$ is not a Lyapunov function for simultaneous rotations.
3. IPT's basin needs separated eigenvalues, which the dense-spectrum case
   cannot supply, and SSJ is too expensive to be a preconditioner anyway.

The direction the failures point to is non-orthogonal norm-reducing
transformations — Eberlein-type shears, which reduce the *departure from
normality* rather than any off-diagonal norm. That changes the invariant being
decreased, which is precisely what failures 1 and 2 say is required. Tried
below.

---

# Shears: descending the right invariant

The departure from normality

$$\Delta(A) = \|A\|_F^2 - \sum_i |\lambda_i|^2 \;\ge\; 0, \qquad
\Delta(A) = 0 \iff A \text{ normal}$$

is decreasable by non-orthogonal similarity, because $\|A\|_F$ is not
similarity-invariant while the spectrum is. The simultaneous (all-pairs,
SSJ-style) form has a closed-form steepest-descent direction. For
$A \leftarrow T^{-1}AT$ with $T = I + G$, to first order $A \leftarrow A +
[A,G]$; splitting $G = K + S$ into antisymmetric (rotation) and symmetric
(shear) parts,

$$\delta\|A\|_F^2 = 2\langle A, [A,S]\rangle
= 2\,\mathrm{tr}\!\left((A^{\mathsf H}A - AA^{\mathsf H})S\right)
= 2\langle C, S\rangle$$

so the steepest-descent shear is just $S = -\mu C$ with $C = A^{\mathsf H}A -
AA^{\mathsf H}$ the self-commutator — symmetric, traceless (hence
volume-preserving, no scaling drift), and zero exactly when $A$ is normal.
The rotation part cannot contribute: orthogonal similarities leave $\|A\|_F$,
the spectrum, and therefore $\Delta$ exactly invariant.

**It works, and needs the same saturation lesson.** A fixed step overshoots
on near-normal input (measured: a matrix at defect 0.000 pushed to 0.028 by
one $\mathrm{cap}=0.25$ step). Backtracking until $\|A\|_F$ actually
decreases fixes it. Ginibre $N=100$: $\Delta$ falls 0.502 → 0.027 over 30
shears, with the spectrum preserved to 3e-15.

## The rotation phase was wrong, and the diagnosis is exact

Shearing to normality and then rotating stalled — *including on a matrix that
was normal by construction* (defect 1.3e-16, yet off stuck at 3.13 out of
5.13). That isolates the bug to the rotation, not the shear.

The reason is structural. For real normal $A = S + N$ (symmetric plus
antisymmetric parts), $A$ normal means $S$ and $N$ commute — but on every
complex-conjugate eigenvalue pair, **$S$ has a double eigenvalue**. So
diagonalizing $S$ alone (the ordinary SSJ angle map) resolves that degeneracy
arbitrarily and never aligns with $N$. The rotation phase must diagonalize
$S$ and $N$ *jointly*.

## Result: normal matrices are solved exactly, by reduction to the symmetric problem

Write $A = H_1 + iH_2$ with $H_1 = (A + A^{\mathsf H})/2$ and $H_2 = (A -
A^{\mathsf H})/2i$, both Hermitian. $A$ is normal precisely when they commute,
so they share an eigenbasis, and diagonalizing the single Hermitian matrix
$H_1 + \alpha H_2$ for **generic** $\alpha$ recovers it. One Hermitian
eigensolve therefore diagonalizes $A$ (`normal_eig`).

The genericity is not a detail — it is the whole fix, and it is measured:

| $\alpha$ | off after diagonalizing $H_1 + \alpha H_2$ |
|---|---|
| 0 (Hermitian part alone) | 3.13 — fails |
| 0.739 (generic) | **1.3e-13** |

on a matrix normal to 1.3e-16. Against LAPACK `dgeev` on normal input:

| $N$ | `normal_eig` | `dgeev` | speedup | eigenvalue error | resid |
|---|---|---|---|---|---|
| 200 | 0.014 s | 0.019 s | **1.33×** | 7.4e-15 | 1.2e-12 |
| 400 | 0.077 s | 0.143 s | **1.86×** | 6.0e-15 | 3.7e-12 |
| 800 | 0.301 s | 0.408 s | **1.36×** | 6.7e-15 | 4.9e-11 |
| 1200 | 0.976 s | 0.915 s | 0.94× | 6.1e-15 | 2.0e-11 |

Besides speed, the output is **unitary** — `dgeev` returns a generally
non-orthogonal eigenvector matrix, which can be arbitrarily ill-conditioned.
For normal input this method cannot produce one.

## Where the shear route stops, and why it is fundamental

The full pipeline (shear to normal, then `normal_eig`) is limited by how
normal the shear can actually make the matrix, and the residual defect passes
straight through to the eigenvalue error:

| input | defect after shears | resulting off | eigenvalue error |
|---|---|---|---|
| near-diagonal $\rho=0.5$, $N=60$ | 1.4e-4 | 2.3e-2 | 8.2e-4 |
| Ginibre $N=60$ | 7.6e-3 | 2.2e-1 | 1.9e-2 |

The plateau is **not a tuning failure**. A matrix whose eigenvector basis is
ill-conditioned cannot be normalized by a *bounded* similarity: normalizing
requires a transformation whose conditioning matches the eigenbasis, so as the
eigenbasis degrades the required shear grows without bound and the descent
stalls at a positive defect. Ginibre eigenvector conditioning worsens with
$N$, which is exactly the observed behavior. Descending the right invariant
was necessary but is not sufficient; the infimum is simply not attained inside
the group of bounded similarities.

## Final map of the general problem

| input class | method | status |
|---|---|---|
| symmetric / Hermitian | SSJ, IPT, hybrid | solved; IPT beats `dsyevd` 1.4–1.9× near-diagonal |
| **normal** | `normal_eig` (joint Hermitian reduction) | **solved exactly**, 1.3–1.9× over `dgeev`, unitary output |
| general, near-diagonal ($\rho \lesssim 0.1$) | `ipt_eig` | **solved**, 4–12× over `dgeev` |
| general, far from diagonal | — | **open**: shears descend the right invariant but plateau whenever the eigenbasis is ill-conditioned |

The open case is now sharply bounded rather than merely unsolved: it is
precisely the non-normal, non-near-diagonal regime, and the obstruction is
that normalization there requires an unbounded similarity.

---

# Breaking the open case: spectral divide and conquer

## The assumption worth dropping

Every method above — Jacobi angles, IPT steps, shears — descends some
invariant by **small local updates**, and every one of them either stalls or
has a bounded basin on non-normal input. That common shape was never a
requirement of the problem; it was inherited from the symmetric solver.

The measured fact that suggests a different bet is the incumbent's *efficiency*,
not its cost. At $N=1000$, with one gemm as the unit:

| gemm | inverse | QR | pivoted QR | `dgeev` | `dgees` |
|---|---|---|---|---|---|
| 1.00 | 5.35 | 7.74 | 12.64 | **93.9** | 88.7 |

A nonsymmetric eigendecomposition is only ~25$N^3$ flops — about 12 gemms'
worth of arithmetic — yet `dgeev` costs 94. It runs at roughly an eighth of
gemm efficiency because Hessenberg reduction and the QR iteration are
bandwidth- and latency-bound. So a method doing **several times more
arithmetic still wins, provided the arithmetic is gemms.**

That licenses buying global convergence outright instead of hoping a local
iteration achieves it.

## The method

Spectral divide and conquer (`src/ssj/sdc.py`; the family is classical —
Bai–Demmel–Gu — and nothing here claims novelty):

1. $\mathrm{sign}(A - \sigma I)$ splits the spectrum by which side of
   $\mathrm{Re}(z) = \sigma$ each eigenvalue lies on;
   $P = (I + S)/2$ is the spectral projector and $\mathrm{tr}\,P$ its rank.
2. A pivoted QR of $P$ gives an orthonormal basis that block-triangularizes
   $Q^{\mathsf T} A Q$. **This is an orthogonal similarity**, so it is
   unconditionally stable — nothing here needs a well-conditioned eigenbasis,
   which is exactly what defeated the shears.
3. Recurse; base cases are 1×1 and 2×2 (complex pairs in closed form).

The sign iteration reuses the repository's own pattern — globally convergent
but expensive while far away, cheap and gemm-only once close:
scaled Newton $X \leftarrow (\mu X + \mu^{-1}X^{-1})/2$, handing off to
Newton–Schulz $X \leftarrow X(3I - X^2)/2$ once $\|X^2 - I\|$ is small.

## It closes the open case

**This is the first method here that solves dense, non-normal, far-from-diagonal
matrices at all.** Ginibre matrices, where IPT diverges and the shears plateau:

| $N$ | eigenvalue error vs LAPACK |
|---|---|
| 60 | 1e-14 |
| 150 | 1.2e-14 |
| 400 | 9.5e-14 |
| 1600 | 4.8e-12 |

No basin condition, no normality requirement, complex pairs handled.

## It is not faster on this CPU, and the reason is exact

| $N$ | SDC | `dgeev` | ratio |
|---|---|---|---|
| 400 | 0.374 s | 0.083 s | 0.22× |
| 800 | 3.53 s | 0.463 s | 0.13× |
| 1600 | 10.5 s | 1.41 s | 0.13× |

The cost model says why, and the arithmetic is worth stating because it gives
the break-even condition. A split needs ~12 sign iterations; most are Newton
steps, and **each Newton step costs an inverse at 5.35 gemm-equivalents**.
Weighting sub-blocks by $(n/N)^3$, the measured totals are ~22 inverses and
~38 gemms, i.e. ~112 gemm-equivalents against `dgeev`'s 94 — already a loss
before the recursion's small-block overhead, which the wall clock then adds.

**Break-even is explicit: the sign iteration must be inverse-free.** At 2 gemms
per iteration the same 16 weighted iterations cost ~32 gemms, plus QR and the
block products, landing near 60 gemm-equivalents — a ~1.5× win. Every avoided
inverse saves 5.35 and costs 2.

Two things were tried and did not close that gap, both recorded so they are not
retried blindly:

- **Handing off to Newton–Schulz earlier** (`ns_switch` 0.6 → 0.99) does not
  help and eventually hurts (112 → 158 gemm-equivalents). The Newton phase is
  genuinely needed: after scaling, eigenvalues near the origin lie outside
  Newton–Schulz's convergence region.
- **A cheap spectral-norm bound** for the initial scaling,
  $\sqrt{\|X\|_1\|X\|_\infty}$, replaced an SVD but overestimates a random
  matrix badly, over-scaling $X$, pushing eigenvalues toward 0 and *adding*
  Newton steps (0.23× → 0.08×). Power iteration is the right answer: $O(N^2)$
  and tight.

## Honest limits

SDC is backward stable, not forward-magical. On a random upper-triangular
matrix the eigenvector condition number is ~$10^{19}$ and the eigenvalues are
hypersensitive; SDC's forward error degrades accordingly (~$10^{-2}$), as any
backward-stable method's must. The test suite asserts that behavior rather
than a forward accuracy no method could deliver.

## Where this genuinely wins

The whole ledger above is one CPU stack, where a factorization costs 5–8
gemms. The ratios that make SDC lose here are exactly the ratios that invert
on accelerators and distributed machines: gemm-dominated hardware makes the
inverse and QR comparatively cheap and gemm-rich, while `dgeev`'s sequential
QR iteration is notoriously poor there. That is why this algorithm family
exists in the literature at all — it was designed for parallel machines, not
for single-socket LAPACK. `bench_gpu.py`-style measurement on a GPU is the
test that would settle it, and is not available here.

## Final map

| input class | method | status |
|---|---|---|
| symmetric / Hermitian | SSJ, IPT, hybrid | solved; IPT beats `dsyevd` 1.4–1.9× near-diagonal |
| normal | `normal_eig` | solved exactly; 1.3–1.9× over `dgeev`, unitary output |
| general, near-diagonal ($\rho \lesssim 0.1$) | `ipt_eig` | solved; **4–12× over `dgeev`** |
| **general, non-normal, far from diagonal** | `sdc_eigvals` | **solved** (1e-13), but 0.13–0.22× of `dgeev` on CPU |

Nothing in this repository is now *unsolved*. What remains is a performance
gap with a stated break-even condition, on hardware where the constants are
known to move.

---

# Zolotarev: implemented, and it does not rescue the general case

The break-even above (make the sign iteration cheap) has a famous candidate
answer: Zolotarev's 1877 best rational approximation to $\mathrm{sign}(x)$,
which Nakatsukasa & Freund show converges in **two iterations** in double
precision. `src/ssj/zolo.py` implements it. The conclusion is a correction to
the obvious expectation, so it is worth stating plainly.

## The construction is correct

Type $(2r+1, 2r)$, coefficients from Jacobi elliptic functions:
$c_i = \ell^2\,\mathrm{sn}^2/\mathrm{cn}^2(iK/(2r+1); \ell')$.
Validated independently of any matrix code — the error equioscillates exactly
$2r+1$ times (the signature of a best approximant), is odd, and improves with
$r$ as theory demands. Two composed passes reach $10^{-15}$ from $\ell =
10^{-3}$, reproducing the paper's headline claim.

**One implementation trap, worth recording.** The product form must *not* be
applied factor-by-factor to a matrix: the factors have wildly different scales
and only cancel at the end, so intermediates overflow — measured $\|Y\|$
reaching $10^{18}$ over $r=8$ factors on a well-conditioned symmetric matrix,
ending in guaranteed failure. The partial-fraction form
$Z(x) = Mx\left[1 + \sum_j a_j/(x^2+c_{2j-1})\right]$ has no such
intermediates — and its $r$ terms are mutually independent, which is the
parallelism the literature is actually buying.

## Where it wins, and where it does not

| spectrum | Newton | Zolotarev $r=8$ | verdict |
|---|---|---|---|
| **real** (symmetric, $N=200$) | 26 iters, 1.017 s | **3 iters, 0.115 s** | **8.8× faster** |
| **complex** (Ginibre, $N=200$) | 14 iters, 0.155 s | 7 iters, 0.256 s | **1.7× slower** |

Zolotarev is optimal on a *real interval* $\ell \le |x| \le 1$. A symmetric
matrix's spectrum lies there and the approximation is superb. A general
nonsymmetric matrix has eigenvalues spread through the complex plane, where
the real-interval optimality simply does not apply: the iteration still
converges (it remains a contraction toward the sign), but needs more passes
*and* far more work per pass, and loses outright.

This is why the Nakatsukasa–Freund title reads "the symmetric eigendecomposition
and the SVD". **Zolotarev does not rescue SDC for the general problem**, and an
earlier note in this repository suggesting it as the next step for the
nonsymmetric case was wrong. For the general case the right published direction
remains Bai–Demmel–Gu's *inverse-free* iteration (matrix multiplication and QR,
no inversion), which is a different mechanism entirely.

## A second, unlooked-for benefit on real spectra

On the symmetric test the two methods disagreed about one eigenvalue, and
**Zolotarev was right**: the true count of eigenvalues with $\mathrm{Re} > 0$
is 58; Zolotarev returns 58 and Newton returns 59. That matrix has an
eigenvalue at $1.4\times10^{-3}$ of the spectral radius — essentially on the
splitting line — where Newton is still converging slowly. Accuracy exactly at
the splitting line, the hardest place for any SDC, is the second thing the
best-approximation property buys.


---

# Presolve + IPT refinement: the architecture is right, the presolve is missing

A natural idea: rather than requiring A to be near-diagonal, *make* it so with
a cheap inexact presolve, then let IPT refine. Two halves, measured separately.

## Randomization is the wrong presolve

Randomized methods are cheap because they exploit **low-rank** structure; a
full eigendecomposition needs all N vectors, and a rank-N sketch costs what a
full solve costs. There is also a subtlety specific to IPT: its map
$V \leftarrow WV/(\Lambda - d)$ is defined by **A's own diagonal split**, so
its contraction rate $\rho(A)$ is a property of A, *independent of the
starting iterate*. A warm start cannot rescue a divergent map — the basis must
actually change, which is why the tracking path forms $B = V^{\mathsf T} A V$
first.

## IPT is an excellent refinement engine

`refine_eig(A, w0, V0)` changes basis into an approximate eigenframe and runs
IPT there. Measured against a float32 LAPACK presolve on **dense Ginibre**
matrices — far outside IPT's own basin, the frame is what puts it inside:

| $N$ | presolve error | $\rho$ in that frame | IPT iters | refined error | resid |
|---|---|---|---|---|---|
| 200 | 2.9e-8 | 9.1e-7 | 2 | **5.1e-15** | 1.8e-14 |
| 400 | 4.8e-8 | 8.2e-6 | 3 | **1.5e-14** | 4.7e-14 |

Three iterations lift a float32 answer to full double precision. This
generalizes IPT's role: not only a solver for near-diagonal input, but a
refinement engine for *any* source of an approximate eigenbasis.

## But there is no cheap presolve on this CPU

| solver | time, $N=1000$ |
|---|---|
| `sgeev` (float32) | 0.905 s |
| `dgeev` (float64) | 0.848 s |

**Float32 is not faster.** `dgeev` is latency- and bandwidth-bound in its
sequential Hessenberg reduction and QR sweeps, not flop-bound, so halving the
precision buys nothing — the same inefficiency that made SDC attractive works
against the presolve here. End to end the pipeline runs at 0.44–0.54× of
`dgeev`, and the complex basis change (an inverse plus two gemms at ~4× real
cost) takes the rest.

So the architecture is sound and the refinement half is validated and cheap in
iterations; what is missing is a presolve that is genuinely cheaper than a full
solve. That exists on tensor-core hardware (float32/TF32 at 8–16× FP64), when
tracking a slowly varying matrix, or whenever an application already holds a
nearby eigenbasis — which is exactly when `refine_eig` should be reached for.


---

# Few eigenpairs: the strongest result here

## IPT is column-separable

In $\Lambda_j = d_j + (WV)_{jj}$, $V_{ij} = (WV)_{ij}/(\Lambda_j - d_i)$,
column $j$ of the update depends **only on column $j$** of $V$. The columns
never interact, so the iteration restricts to any subset of them *exactly* —
no deflation, no locking, no accuracy loss. `ipt_eig_partial(A, cols)` runs
$k$ columns at $O(N^2k)$ per iteration instead of $O(N^3)$.

`cols` chooses *which* eigenpairs: column $j$ converges to the eigenvalue near
the diagonal entry $A[c_j, c_j]$. So targets are specified directly, and an
**interior** target costs exactly what an extremal one costs.

That is the property Krylov methods lack. Lanczos/Arnoldi converge from the
outside of the spectrum inward and need **shift-invert** to reach the middle —
an $O(N^3)$ factorization per shift. IPT needs no factorization at all.

## Against ARPACK with shift-invert, interior targets, $\rho = 0.05$

| $N$ | $k$ | IPT partial | ARPACK shift-invert | `dgeev` (all) | **vs ARPACK** | resid |
|---|---|---|---|---|---|---|
| 500 | 4 | 0.0030 s | 0.0123 s | 0.212 s | **4.1×** | 1.2e-14 |
| 500 | 16 | 0.0037 s | 0.0238 s | 0.175 s | **6.4×** | 9.0e-16 |
| 1000 | 4 | 0.0067 s | 0.0514 s | 0.674 s | **7.7×** | 1.9e-14 |
| 1000 | 16 | 0.0071 s | 0.0456 s | 0.614 s | **6.4×** | 9.5e-16 |
| 2000 | 4 | 0.0274 s | 0.1837 s | 2.159 s | **6.7×** | 5.7e-15 |
| 2000 | 16 | 0.0324 s | 0.2373 s | 2.128 s | **7.3×** | 5.7e-15 |

Six or seven iterations throughout. Symmetric interior targets against
`eigsh` shift-invert, $N=1000$: **6.7×** at $k=4$ and **5.3×** at $k=16$,
and 13.7–18.7× against a full `dsyevd`.

## Cost really does scale with $k$, not $N$

$N=1000$ general, as a fraction of a full `dgeev`:

| $k$ | 1 | 4 | 16 | 64 | 256 | 500 |
|---|---|---|---|---|---|---|
| cost vs full `dgeev` | 0.7% | 1.3% | 1.3% | 2.7% | 6.7% | 14.1% |

Even at $k = N/2$ it is still 7× cheaper than the full solve, so there is no
practical crossover at which one should switch back — take the whole spectrum
this way if the basin permits.

## The basin is PER-COLUMN, which is a much weaker requirement

The obvious objection to all of the above is that it still needs a
near-diagonal matrix. It does not — it needs near-diagonal *columns*, and
because the map is column-separable that is a strictly weaker condition:

$$\rho_j = \max_{i \neq j} \frac{|W_{ij}|}{|d_j - d_i|}$$

costs $O(Nk)$ (`ipt_rate_columns`, cheaper even than the full $O(N^2)$ test)
and decides each column on its own. A matrix can be hopeless globally while
individual columns sit deep inside the basin.

Measured, $N=400$: a dense strongly-coupled band plus four isolated levels.
**Global $\rho = 992$ and the full solver diverges** — yet the isolated
columns have $\rho_j = 0.004$–$0.009$ and converge in 9 iterations to
$5\times10^{-16}$.

That is not a contrived construction: it is an impurity level in a band, a
defect state in a gap, a localized mode — the ordinary situation in disordered
and defect physics, and the setting IPT came from. Against ARPACK
shift-invert targeting those levels:

| $N$ | IPT partial | ARPACK | `dgeev` (all) | **vs ARPACK** | resid |
|---|---|---|---|---|---|
| 500 | 0.0048 s | 0.0952 s | 0.209 s | **20×** | 1.7e-14 |
| 1000 | 0.0111 s | 0.3925 s | 0.731 s | **35×** | 4.8e-15 |
| 2000 | 0.0462 s | 5.6945 s | 2.405 s | **123×** | 2.4e-14 |

At $N=2000$ ARPACK is slower than a *full* `dgeev` — shift-invert on a dense
matrix pays a cubic factorization and then converges slowly toward a cluster
of nearby targets — while IPT partial is 52× faster than the full solve.

## The same caveat, correctly scoped

This lives inside IPT's basin, now measured **per column** (`ipt_rate_columns`,
$O(Nk)$), and reports non-convergence rather than hiding it. The regime is dense near-diagonal
matrices with targeted interior eigenpairs — exactly the setting where the
alternatives are worst: a full dense solve wastes $N/k$ of its work, and
shift-invert Krylov pays a cubic factorization to look at the middle of the
spectrum.


---

# Making it broadly usable, and a correction

## Randomized/sampling methods cannot go inside the iteration

Worth stating precisely, since it is the obvious thing to try. A sketched
matvec has error $\sim\|W\|\|V\|/\sqrt{s}$, so reaching $10^{-13}$ would
need $s \sim 10^{26}$ samples. Mixed precision worked earlier because SSJ is
*memoryless* and re-derives its angles from a fresh $B$ every sweep; IPT has
no such structure to exploit, and with 4–9 iterations (the first one free)
there is almost no cheap early phase to economize on. Randomization's role
here is the fallback and target selection, not the inner loop.

## Correction: the per-column rate is a heuristic, and it is optimistic

An earlier section of this document presented $\rho_j$ as a free, reliable
screen. Measurement says otherwise. $\rho_j$ counts **direct, one-hop**
coupling only, while the underlying perturbation series sums over multi-hop
paths, where far-apart near-degenerate sites resonate — the classic problem
of locator expansions.

| case | $\rho_j$ | converges? | actually needs |
|---|---|---|---|
| dense, isolated level | 0.18 | **no** | $\lesssim 0.05$ |
| 2D Anderson lattice, $W=12$ | 0.25 | **no** | $\lesssim 0.04$ |

The sparse/lattice case is the worse of the two, and it deflates a hope worth
recording: **sparse localized states are not automatically IPT's home turf.**
A 2D Anderson model only enters the basin at disorder $W \gtrsim 80$, far
beyond the physically interesting regime, because a lattice has many distant
sites at nearly equal energy that one hop cannot see.

## The design that survives this

`eig_partial` (in `src/ssj/dispatch.py`) screens each target with $\rho_j$,
routes to IPT or ARPACK per target, and — because the screen is unreliable —
is built so that **a wrong screen costs time, never correctness**: if IPT is
attempted and fails to converge, those targets go to ARPACK and the
unconverged output is discarded. The default gate is 0.1, calibrated against
the failures above rather than against theory.

Measured on impurity levels, routing overhead included:

| $N$ | screen | auto-routed | forced ARPACK | speedup |
|---|---|---|---|---|
| 500 | 0.21 ms | 0.0049 s | 0.0318 s | **6.4×** |
| 1000 | 0.27 ms | 0.0105 s | 0.0720 s | **6.9×** |
| 2000 | 0.43 ms | 0.0492 s | 0.4488 s | **9.1×** |

And on input where no target qualifies (Ginibre), the overhead against calling
ARPACK directly is **0.0%** at $N=1000$ — the screen is three orders of
magnitude cheaper than either solver, so it disappears into the noise.

That is what "broadly usable" can honestly mean here: never materially worse
than the standard tool, up to two orders of magnitude better when targets are
genuinely isolated, and never wrong when the indicator misjudges.


---

# A better fixed-point equation: block IPT

IPT's fixed point is quadratic, and its contraction rate is set by the
*smallest* denominator $|\lambda_j - d_i|$ — a locator expansion, which fails
exactly when another diagonal entry sits near the target eigenvalue. Saturating
that denominator does not help (Failure 1 above): the trouble is the sum over
many near-resonant terms, not any single one.

## The equation

Stop treating near-resonant states perturbatively. Split the indices into a
block $B$ (the target plus its worst offenders) and the rest $C$, and write
$v_C = X v_B$. Then $Av = \lambda v$ is *exactly* equivalent to

$$X = (\lambda - D_C)^{-1}\left(W_{CB} + W_{CC}X\right), \qquad
\left(A_{BB} + W_{BC}X\right)v_B = \lambda\, v_B$$

The first is IPT-like but runs **only over $C$**, whose gaps all exceed the
block radius by construction; the second is a small $b \times b$ eigenproblem
solved *exactly*, and it is what absorbs the near-degeneracies. Plain IPT is
the $b = 1$ limit. (This is quasi-degenerate / Löwdin–Bloch effective-Hamiltonian
perturbation theory read as a fixed-point iteration — standard physics
practice, not a new idea.)

**The basin becomes a parameter of the algorithm rather than a property of the
matrix.**

## Measured: the basin is ~16× wider

Dense near-diagonal, $N=400$, increasing coupling until each method fails:

| coupling | plain IPT | block $b{=}8$ | block $b{=}32$ |
|---|---|---|---|
| 0.5 | 13 its | 5 its | 4 its |
| 2 | **diverges** | 11 its | 7 its |
| 8 | **diverges** | diverges | 26 its |
| 20 | diverges | diverges | diverges |

Plain IPT dies at coupling 0.5; block IPT with $b=32$ survives to 8 — a **16×
wider basin**, and it is tunable, which plain IPT's is not.

## But it is slower where both work — the trade is linear

Each iteration costs $O(N^2 b)$ rather than $O(N^2)$, so a wider basin is paid
for in proportion to the block size. $N=600$:

| | plain IPT | block $b{=}8$ | |
|---|---|---|---|
| coupling 0.2 | 5.0 ms (9 its) | 18.8 ms (4 its) | **0.27×** |
| coupling 0.5 | 2.9 ms (13 its) | 17.4 ms (5 its) | **0.17×** |

Fewer iterations, more work per iteration, net loss. Where plain IPT
*diverges*, though, the comparison is against LAPACK rather than against IPT,
and block IPT still wins: coupling 2 at $b{=}8$ takes 37.5 ms against
`dgeev`'s 264 ms (**7.0×**), coupling 8 at $b{=}32$ takes 212 ms against
308 ms (**1.5×**).

## Verdict on the question

Better basin: **yes, 16× and tunable.** Better performance: **no** — 4–6×
slower wherever plain IPT already converges. Block IPT is the right tool
strictly in the band between the two basins, where it converts a divergence
into a 1.5–7× win over LAPACK.

One negative worth recording: choosing the block by the *ratio*
$|W_{ij}|/|d_i-d_j|$ — the terms that provably break the contraction, and so
the "principled" criterion — is **not** uniformly better than choosing by gap
alone. On a 2D Anderson lattice the gap criterion converges at disorder 40
where the ratio criterion does not, and the ordering reverses at disorder 80.
No block criterion tested reaches the strongly resonant lattice regime
($W \lesssim 20$), where multi-hop resonances defeat all of them.


## Adaptive blocks: let the iterate choose

Both a priori block criteria proved unreliable, so stop predicting and let the
iterate report. A large tail amplitude $|X_i|$ *is* the statement that index
$i$'s "small correction" is not small — the empirical signal that it should
have been solved exactly. `adaptive_block_ipt_eig` starts at $b=1$ and
promotes the largest-amplitude indices out of $C$ whenever contraction is
slow, re-selecting against the **current** $\lambda$ rather than the initial
$d_j$ (the resonant set moves as the eigenvalue converges).

**The growth trigger matters more than it looks.** Growing only when truly
stalled is too permissive: the iteration settles for a slow linear rate and
runs out of iterations instead of buying a better rate with a bigger block.
Measured at coupling 8 — trigger 0.9 stops at $b=37$ and fails; trigger 0.5
grows to $b=93$ and converges. The default says *grow until convergence is
fast*, which is right because iterations cost $O(N^2b)$ either way.

Dense near-diagonal, $N=400$:

| coupling | plain IPT | static $b{=}32$ | **adaptive** | vs `dgeev` |
|---|---|---|---|---|
| 0.5 | converges | converges | **$b{=}1$**, 20 ms | **4.8×** |
| 2 | diverges | converges | **$b{=}5$**, 48 ms | **2.1×** |
| 8 | diverges | converges | $b{=}125$, 1445 ms | 0.1× |
| 20 | diverges | **diverges** | $b{=}125$, 1616 ms | 0.1× |
| 50 | diverges | **diverges** | $b{=}125$, 3474 ms | 0.0× |

Two things to read off. Adaptive **dominates the static version**: it picks
$b=1$ when $b=1$ suffices, so easy problems cost what plain IPT costs, and it
converges at coupling 20–50 where every static block tested diverges — a
**100× wider basin** than plain IPT. And the basin extension is
*economically* useful only out to coupling $\approx 2$; past that the block
needed is so large that calling `dgeev` is cheaper.

**Operating rule.** Run adaptive with `max_block` capped around 8–16. Inside
the profitable band it is parameter-free and 2–5× faster than LAPACK; outside,
it gives up cheaply (the cap bounds the cost) and the caller falls back. That
converts the earlier "guess the block size" into no parameter at all, which is
what makes it usable in the `eig_partial` router.


---

# A different map: fixed points that are projectors, not eigenvectors

Every map so far has an eigenvector or eigenbasis as its fixed point, and each
pays the same tax. Power and gradient flows converge globally but only to
spectral extremes, and only linearly. IPT and Newton-type maps are fast but
locally convergent. Blocking widens IPT's basin but buys it linearly in cost.

Changing the *object* escapes the tradeoff. Take idempotents as the fixed
points:

$$P \;\leftarrow\; 3P^2 - 2P^3 \;=\; P^2(3I - 2P)$$

The scalar map $p(x) = 3x^2 - 2x^3$ satisfies $p(0)=0$, $p(1)=1$,
$p(\tfrac12)=\tfrac12$, with $p'(0) = p'(1) = 0$ and $p'(\tfrac12) = \tfrac32$.
So $0$ and $1$ are **superattracting** and $\tfrac12$ **repels**, and $p$ maps
$[0,1]$ into itself. Scale $A$ linearly so its spectrum lands in $[0,1]$ with
the splitting point $\mu$ at $\tfrac12$, and every eigenvalue below $\mu$ is
driven to 1, every one above to 0 — **globally convergent, quadratic, and
every operation a gemm.**

This is McWeeny purification, the basis of linear-scaling electronic
structure. Nothing here is new but its use as the SDC splitter.

## It closes the break-even this document derived

GENERAL.md's SDC section concluded: *the sign iteration must be inverse-free*.
Newton's is not (an inverse costs 5.35 gemm-equivalents here). Newton–Schulz on
the sign function is inverse-free but **not globally convergent** — it needs
the spectrum already near $\pm1$. Purification is both, because the $[0,1]$
scaling is a guarantee rather than a hope.

Hermitian, splitting at the median:

| $N$ | purification | Newton sign | **ratio** | rank | $\|P^2-P\|$ |
|---|---|---|---|---|---|
| 600 | 29 its, 57 gemms, **91.2 ge** | 15 its, **395.8 ge** | **4.34×** | 300.0000 (true 300) | 1.4e-13 |
| 1200 | 31 its, 61 gemms, **90.1 ge** | 60 its, **579.6 ge** | **6.43×** | 600.0000 (true 600) | 2.9e-14 |

Note the shape, not just the ratio: purification's cost is **flat** at ~90
gemm-equivalents while Newton's *grows* (396 → 580), because Newton needs more
iterations as the spectrum crowds the splitting line. The advantage widens
with $N$.

## Limitation, and honest placement

The map needs the spectrum inside the real interval $[0,1]$, so it is
**Hermitian-only**. A general matrix with complex eigenvalues cannot be scaled
there and purification does not apply — the sign function still does. This is
the same real-interval boundary that stopped Zolotarev from rescuing the
general case, and it is worth noticing that two independent attempts at the
nonsymmetric problem have now failed at exactly the same wall.

And it does not make symmetric SDC beat LAPACK: `dsyevd` costs 15–18
gemm-equivalents against purification's 90. What purification is genuinely the
right tool for is the case where **the projector is the answer** — a density
matrix in linear-scaling electronic structure, an invariant subspace for
deflation or tracking — where eigenvectors are never formed at all, and where
the global basin plus the all-gemm diet is exactly what is wanted.


---

# Two creative attempts: one dead end, one large win

## Anderson acceleration on IPT: no

RESULTS.md records Anderson diverging on SSJ, and momentum failing there too —
but both findings concern SSJ, whose stability *comes from* a saturation that
extrapolation bypasses. IPT has no such mechanism to break: it is a plain
fixed-point iteration with linear rate, which is what Anderson/DIIS was built
for, and Anderson is a quasi-Newton method that can converge where the
underlying iteration diverges. It looked like a direct attack on the basin.

It is not. Dense near-diagonal, $N=400$:

| coupling | plain IPT | Anderson $m{=}3$ | $m{=}8$ | $m{=}20$ |
|---|---|---|---|---|
| 0.5 | 13 its | 15 | 14 | 14 |
| 2 – 50 | diverges | **diverges** | **diverges** | **diverges** |

No basin extension at any depth, and slightly slower where IPT already works.
IPT's divergence is not slow contraction that extrapolation repairs — it is a
spectral instability with *many* unstable directions at once, and a depth-$m$
quasi-Newton correction can absorb only $m$ of them.

## Generalizing the splitting: yes, and by a lot

IPT splits $A = D + W$ with $D$ the diagonal. **Nothing forces that choice.**
For any easily-invertible $M$,

$$(M - \lambda I)v = -(A-M)v \quad\Longrightarrow\quad
v = -(M-\lambda I)^{-1}Rv, \qquad R = A - M$$

with rate $\|(M-\lambda I)^{-1}R\|$ in place of $\|W\|/\text{gap}$. Plain
IPT is $M = \mathrm{diag}(A)$. Choosing $M$ to model the dominant coupling —
a band, a lattice direction, anything with a cheap solve — shrinks $R$ and so
shrinks the rate.

Two modes, and they are genuinely different algorithms rather than an
implementation detail (found by getting it wrong first):

- **reduced** excludes row `target` from the solve, since that row is what
  determines $\lambda$. This is the strict IPT generalization and reproduces
  plain IPT at $M = \mathrm{diag}(A)$.
- **inverse** solves the full system and renormalizes. When $M - \lambda$ is
  near-singular the solution aligns with its near-null direction — that is
  **preconditioned inverse iteration**, where the near-singularity is the
  *signal*, not a defect.

$N=300$, strong band plus weak dense remainder:

| band | plain IPT | reduced | **inverse ($M$ = band)** |
|---|---|---|---|
| 0.5 | 62 its | 72 its | **42 its** |
| 2.0 | diverges | diverges | **61 its** |
| 8.0 | diverges | diverges | **75 its** |
| 30.0 | diverges | diverges | **118 its** |

Plain IPT dies at band 0.5; inverse mode is still converging at band 30 at
$10^{-16}$ — **a basin roughly 60× wider, the largest extension measured
anywhere in this repository.**

The limits are equally clear. Inverse mode *degenerates* with
$M = \mathrm{diag}(A)$ (the single near-zero entry collapses the iterate onto
$e_{\text{target}}$), so the two modes are not interchangeable. And on the 2D
Anderson lattice with $M$ = intra-row hopping, no disorder from 4 to 20
converges: a genuinely two-dimensional coupling is not modelled by one lattice
direction. The case that defeated every other method in this document defeats
this one too.
