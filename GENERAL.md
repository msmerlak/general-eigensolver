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


---

# A zoo of rewritings

Rather than reason about which reformulation of $Av = \lambda v$ should be
better, here are seven, all given the same matrix, target and tolerance
(`experiments_zoo.py`). Symmetric near-diagonal, $N=300$, target mid-spectrum:

| coupling | IPT | +Rayleigh λ | damped β=½ | +Aitken | self-energy(λ) | Richardson |
|---|---|---|---|---|---|---|
| 0.5 | 21 its | 21 its | 48 its | 101 its | diverges | diverges |
| 2 | **div** | div | **123 its** | div | div | div |
| 8+ | div | div | div | div | div | div |

Mostly nothing. Taking $\lambda$ from the Rayleigh quotient instead of the
$j$-th row changes not one iteration. Aitken extrapolation on the iterate
sequence costs 5× and buys nothing. The scalar self-consistent second-order
self-energy $\lambda = d_j + \sum_k |W_{jk}|^2/(\lambda - d_k)$ — appealingly
cheap at $O(N)$ per step — diverges even where IPT converges, because the poles
at every $d_k$ make it wildly non-monotone. Plain Richardson never had a chance
at an interior target.

**Damping is the one cheap win**: $v \leftarrow v + \beta(T(v)-v)$ with
$\beta = \tfrac12$ converges at coupling 2 where undamped IPT diverges, at the
price of ~2× the iterations where both work. One line of code for a 4× basin.

## The rewriting that actually worked: Davidson

The winner is not a rearrangement of the *formula* but of what is done with it.
IPT applies the diagonal resolvent to the whole vector and **replaces** the
iterate. Davidson applies the identical resolvent to the **residual only**,

$$t = (\lambda I - D)^{-1}\left(Au - \lambda u\right)$$

then appends $t$ to a subspace and takes the best vector available by
Rayleigh–Ritz. Same preconditioner; the correction **accumulates instead of
replacing**. The near-degenerate levels that defeat IPT — because their terms
dominate the perturbation sum — are simply resolved by the small Rayleigh–Ritz
eigenproblem. It is block IPT's medicine, with the block built automatically
from whatever directions the residual explores.

| coupling | IPT | **Davidson** | eig err | resid |
|---|---|---|---|---|
| 0.5 | 21 its | **14 its** | 7.6e-16 | 6.3e-14 |
| 2 | **diverges** | **29 its** | 1.9e-16 | 8.0e-14 |
| 8 | **diverges** | **96 its** | 3.6e-16 | 5.3e-14 |
| 30 | diverges | diverges | (2.6e-5) | 1.1e-1 |

**~16× wider basin than IPT, and faster inside it** — the first variant here
that improves both at once, where blocking bought basin only in linear
proportion to cost.

Two mistakes worth recording, both of which first made Davidson look like a
failure. The correction must be reorthogonalized **twice** — a single
Gram–Schmidt pass loses orthogonality as the subspace grows. And the residual
test must be **relative**: with $\|A\| \approx 300$ an absolute $10^{-12}$
threshold reported divergence on runs that had reached machine precision
(measured eigenvalue error 9.4e-17 on a "failed" run). Both were harness bugs
masquerading as negative results, which is the failure mode this whole survey
is most exposed to.


## Taking Davidson further: where it stops

Davidson's wider basin looked like the way to extend this repository's best
result — `ipt_eig_partial` beats ARPACK shift-invert by 4–123× on interior
targets but only inside IPT's basin, and outside it the router pays for the
O(N³) factorization the whole approach exists to avoid. Davidson keeps the
factorization-free property with a 16× wider basin, so it should widen the
niche. It does not, and the reason corrects my account of why IPT won.

$k=4$ interior targets, $N=400$:

| coupling | IPT partial | $k\times$ Davidson | ARPACK | **D vs ARPACK** |
|---|---|---|---|---|
| 0.5 | 15 its, **3 ms** | 52 its, 95 ms | 9.2 ms | **0.10×** |
| 2 | 149 its, 17 ms | 163 its, 155 ms | 7.3 ms | **0.05×** |
| 8 | **diverges** | 465 its, 561 ms | 7.1 ms | **0.01×** |
| 30 | diverges | diverges | 9.0 ms | — |

Davidson does reach coupling 8 where IPT dies — the basin claim holds — but at
10–100× the cost of simply calling ARPACK.

**So the partial solver's advantage was never "no factorization".** It was
IPT's *iteration count*: 5–15 matvecs, total. ARPACK's shift-invert
factorization is O(N³/3), which at N=400 is perfectly affordable and buys very
fast Krylov convergence; beating it requires finishing in a handful of
matvecs, not merely avoiding a factorization. Davidson needs 50–500, each
O(N²), and that is enough to lose outright.

A block Davidson was also written and discarded rather than shipped: it failed
at coupling 2 where the single-vector version converges, and was already 0.3×
ARPACK where it did work — two independent reasons not to keep it.

The honest boundary of the whole partial-solver result is therefore narrower
than the basin discussion suggests. It is not "wherever a factorization is
expensive"; it is **wherever IPT converges in a handful of iterations**, which
is the near-diagonal regime the per-column screen already identifies, and the
existing router already handles correctly.


---

# A broader campaign: what else was tried

Systematic sweep beyond the zoo above, each measured against a real
benchmark and kept only if it won. `python3 tests/test_window.py` reproduces
the shipped result; the rest are recorded here because they are negative but
non-obvious.

## Windowed purification: the strongest new capability in this repository

Every partial solver here — `ipt_eig_partial`, `davidson_eig`, `block_ipt_eig`
— needs a diagonal entry that already approximates the wanted eigenvalue: they
are targeted solvers for near-diagonal or structured input. Purification
(`ssj.purify`) needs none of that: it computes a spectral projector for ANY
Hermitian matrix, globally, with no basin condition. Two projectors compose
into a **window projector**,

$$P_{\mathrm{window}} = P(\mathrm{hi}) - P(\mathrm{lo})$$

idempotent because both factors are polynomials in $A$ and hence commute
exactly, with rank equal to the exact number of eigenvalues in $[\mathrm{lo},
\mathrm{hi}]$. Extract an orthonormal basis for its range (pivoted QR),
Rayleigh–Ritz on that small basis: exact eigenpairs in the window.

**The reason this earns a place is not speed.** Measured, GOE $N=400/800$: the
count is exact and residuals sit at machine precision ($3.9\times10^{-16}$ to
$1.3\times10^{-15}$), but wall time is **6–7× slower than ARPACK even when
ARPACK is handed the exact count in advance**, and at $N=800$ slower than a
full `dsyevd`. The reason to use it is that the exact count is *needed*, not
guessed. ARPACK shift-invert requires $k$ up front and returns exactly $k$
Ritz values with no warning if the true count is larger. Measured, GOE
$N=600$, a window holding 24 true eigenvalues:

| $k$ guessed | ARPACK returns | actually in-window | |
|---|---|---|---|
| 12 (half) | 12 | 12 | **missed 12, silently** |
| 22 (close) | 22 | 22 | **missed 2, silently** |
| 24 (exact) | 24 | 24 | correct |
| `window_eig` | — no guess needed — | **24** | exact, every time |

Edge cases hold: empty windows (above/below the spectrum) return count 0; a
window covering the whole spectrum returns $N$; three exactly degenerate
eigenvalues inside a window are returned as 3, residual $1.9\times10^{-14}$.
A boundary landing exactly on an eigenvalue is ambiguous by construction, true
of any spectral-projector method and not a defect here.

Use this whenever completeness on an *unknown* count matters more than wall
time: counting states in a spectral window, verifying nothing was missed, or
whenever the honest alternative is guessing $k$ and hoping.

## Jacobi-Davidson: no improvement over plain Davidson, two variants tried

Plain `davidson_eig` already won 16× basin and higher speed over IPT
(previous section). Jacobi–Davidson is the standard next step: solve the
*projected* correction equation $P(A-\theta I)P\,t = -Pr$, $P = I - uu^{\mathsf
T}$, rather than preconditioning the raw residual.

**Diagonal-preconditioned JD** (solve the projected system with $K =
\mathrm{diag}(A)-\theta I$ via the standard rank-one-correction formula, same
$O(N)$ cost per iteration as plain Davidson) changed **nothing**: 14 vs 14,
29 vs 30, 96 vs 103 iterations across the same coupling sweep — noise, not
signal.

**Inner-PCG JD** (a few steps of preconditioned CG on the actual projected
operator, i.e. genuinely solving the correction equation rather than applying
the preconditioner once) reduces *outer* iterations but not total cost:
coupling 2 dropped from 29 outer iterations to 13, but at ~9 matvecs each
($P(A-\theta I)P$ applied via two matrix–vector products per CG step) —
**~120 total matvecs against plain Davidson's 29.** At coupling 8 it's worse
in both counts (891 matvecs vs 96). Plain Davidson is Pareto-better than
either JD variant on this problem class; the correction-equation machinery
buys nothing that the diagonal preconditioner didn't already capture.

## Davidson's basin is anchored to the diagonal — Krylov methods are not

Testing Davidson against `scipy.sparse.linalg.lobpcg` and `eigsh` on
**extremal** eigenvalues (LOBPCG's actual design target) rather than interior
ones exposed a boundary worth stating plainly. $N=400$, target = smallest
eigenvalue:

| coupling | Davidson | LOBPCG | ARPACK (SA) |
|---|---|---|---|
| 0.5 | 12 its, 0.029 s | 0.090 s | 0.027 s |
| 2 | 25 its, 0.032 s | 0.111 s | 0.022 s |
| 8 | 72 its, 0.077 s | 0.105 s (only 2.6e-10) | 0.015 s |
| 30 | **diverges** | 0.102 s, 1.6e-15 | **0.013 s, 1.8e-15** |

At coupling 30 Davidson **diverges even on an extremal target**, while ARPACK
converges every time, faster than everything else in the table. The reason:
Lanczos/Arnoldi Krylov iteration is naturally globally convergent toward
extremal eigenvalues — it needs no starting guess anchored to a diagonal
entry — while Davidson's initial vector $e_j$ is only a good starting point
when the diagonal entry $d_j$ is close to the true eigenvalue, which strong
coupling destroys regardless of whether the target is extremal or interior.

**This sharpens the whole family's niche.** IPT/Davidson/block-IPT own
*interior eigenvalues of near-diagonal-ish matrices*; ARPACK owns *extremal
eigenvalues of anything*; window purification owns *reliable counts of
anything, at a real cost in speed*. None of them is a general replacement for
the others, and this campaign is what located the boundaries precisely rather
than leaving them assumed.

## Full-spectrum purification-based SDC: confirmed non-competitive

Composing `purify_split` recursively into a full Hermitian eigensolver
(the natural next step after the earlier single-split benchmark) was
measured end to end rather than assumed: $N=400/800$, `min_block`
$32$–$128$, **0.04–0.33× of `dsyevd`**, accurate to $8$–$14\times10^{-15}$.
This confirms rather than contradicts the earlier single-split finding
(purification wins on gemm-equivalents against Newton's sign function, but
`dsyevd` at 15–18 gemm-equivalents undercuts both) — not shipped as a new
module since it duplicates a result already established, but worth recording
that the full assembly was actually built and measured, not inferred.


## Two more, both from this same campaign

**Mixed precision for purification: shipped, modest and size-dependent.**
`spectral_projector`/`window_eig` now take `precision="mixed"`, running early
sweeps in float32 (same memoryless mechanism as SSJ's mixed precision — each
iteration recomputes $P^2$ from the *current* $P$, so low precision can only
degrade the warm start, never the final answer). Measured on the raw
projector: 1.43–1.45× at $N=600/1200$, identical rank and idempotency.
End-to-end inside `window_eig` the win is size-dependent — **0.91×** (slightly
slower) at $N=400$, **1.73×** at $N=800$ — because float32's advantage is a
constant-factor cubic effect while per-call overhead is fixed; small problems
don't clear that bar. Worth it past a few hundred, not below.

**Chebyshev-filtered subspace iteration: explored, and a self-correction.**
Polynomial filtering (apply a Jackson-damped Chebyshev approximation to the
window indicator, matvec-only, no matrix squaring) looked like it might beat
purification outright — an early measurement showed 4–6.5× faster at matched
block size and degree. That measurement was comparing at *mismatched*
accuracy. Two real bugs surfaced in getting there, both worth recording:

1. Naive spectral-bound estimation matters enormously. Gershgorin bounds were
   measured $6.8\times$ wider than the true spectral range on GOE, squeezing
   the window into 1.15% of the mapped $[-1,1]$ interval — far beyond what a
   degree-30 polynomial can resolve, which is what caused the *first* failure
   (0 eigenvalues found, looked like the whole idea was broken).
2. Plain power iteration for tight bounds has its own trap: it converges to
   the eigenvalue of *largest magnitude*, not the algebraic max, so on a
   symmetric-around-zero spectrum like GOE both `power(A)` and `power(-A)`
   converged to the *same* (most negative) eigenvalue, collapsing the
   estimated range to a near-empty sliver. Lanczos (a small Krylov subspace,
   reading both extremes off its tridiagonal projection) fixes this reliably.

With both bugs fixed, the filter does resolve the window — but only slowly
near the boundary. Median residual for in-window states drops nicely with
more refinement passes; the *maximum* residual (boundary states) stays stuck
around $10^{-2}$ for many iterations, because the filter's response is a
smooth, moderate-slope function there, not a sharp cutoff. Reaching residuals
comparable to purification's needs degree $\approx150$ and 5–6 refinement
passes. At that *matched* accuracy:

| $N$ | Chebyshev (matched) | purification | ratio |
|---|---|---|---|
| 400 | 0.371 s, found 40/40, err 3.5e-11 | 0.221 s, found 40/40 | **0.60×** |
| 800 | 1.689 s, **found 7/80** | 1.270 s, found 80/80 | **0.75× (and wrong)** |

**Not shipped.** At $N=800$ the "matched" settings that worked at $N=400$
undercounted catastrophically — exactly the silent-miss failure mode this
whole module exists to eliminate, and worse than the honest 0.6× at $N=400$.
A production version would need adaptive degree/block-size selection to be
trustworthy, which would very likely erode the constant-factor advantage
further rather than recover it. The corrected conclusion: purification's
specific fixed-point structure (superattracting away from its one repelling
point) gives predictable, reliable convergence that a moderate-degree
polynomial filter does not match without cost approaching or exceeding it.


## Last check: does a better STARTING vector fix strong-coupling divergence?

One more test of the extremal-target divergence found above, because it has
an obvious-looking fix. Seed Davidson's initial subspace with a few Lanczos
vectors (which need no diagonal anchor at all) instead of just $e_j$ — if the
problem were merely a bad starting guess, a handful of Krylov directions
should rescue it.

$N=400$, extremal target, seed dimension 1/4/8:

| coupling | seed=1 | seed=4 | seed=8 |
|---|---|---|---|
| 8 | 72 its | 70 its | 69 its |
| 30 | diverges | diverges | diverges |
| 100, 300 | diverges | diverges | diverges |

No rescue at any seed size. This sharpens the diagnosis rather than just
repeating it: the failure is not the *starting point*, it is that Davidson's
diagonal preconditioner $K = \mathrm{diag}(A) - \theta I$ is applied at
**every** correction step, and once coupling dominates the diagonal, $K$ is a
bad model of $A - \theta I$ throughout the whole run, not just at the start —
no amount of exploring around a bad anchor compensates for a preconditioner
that stays wrong. This is exactly the gap `gipt.py`'s generalized splitting
already closes by replacing $\mathrm{diag}(A)$ with a structured $M$ that
actually captures the dominant coupling (measured ~60× basin widening there);
Davidson with a genuinely better $M$ in place of the diagonal, rather than a
better starting vector, is the untried combination this points to next.

## The map, for future work

| capability | function | wins against | measured |
|---|---|---|---|
| symmetric near-diagonal | `ipt_eigh` | `dsyevd` | 1.4–1.9× |
| general near-diagonal | `ipt_eig` | `dgeev` | 4–12× |
| normal (any) | `normal_eig` | `dgeev` | 1.3–1.9×, unitary |
| $k$ eigenpairs, interior, near-diagonal columns | `ipt_eig_partial` | ARPACK shift-invert | 4–123× |
| $k$ eigenpairs, arbitrary input | `eig_partial` | ARPACK (never worse) | 6.4–9.1× when it qualifies |
| single eigenpair, wider basin | `davidson_eig` | `ipt_eig_partial` | 16× basin, faster inside it |
| single eigenpair, resonant/lattice | `block_ipt_eig`, `adaptive_block_ipt_eig` | plain IPT | up to 100× basin |
| single eigenpair, band/block-dominant | `gipt_eig(mode="inverse")` | plain IPT | ~60× basin |
| all eigenpairs in an interval, unknown count | `window_eig` | ARPACK (correctness, not speed) | exact count vs silent misses |
| dense non-normal, no structure | `sdc_eigvals` | nothing else here solves it | 1e-13, 0.13–0.22× `dgeev` |

Dead ends worth not retrying, all measured rather than assumed: Anderson
acceleration on IPT (no basin extension at any depth); Jacobi–Davidson, both
diagonal-projected and inner-PCG variants (no improvement or strictly worse
total cost than plain Davidson); LOBPCG/block-Davidson on the $k$-eigenpair
problem (10–100× *slower* than ARPACK — the win there was never
"no factorization", it was IPT's low iteration count, and anything needing
tens to hundreds of iterations loses that trade); Chebyshev-filtered subspace
iteration for interval eigensolving (looked 4–6× faster at mismatched
accuracy, 0.6–0.75× slower and unreliable at matched accuracy); Lanczos-seeded
Davidson starting vectors (does not fix a preconditioner that is wrong
throughout the run, only ever helps preconditioners that are merely
*starting* wrong).


---

# The largest margin found: large sparse, interior targets

Every earlier partial-solver benchmark here was **dense**, where ARPACK's
shift-invert factorization is $O(N^3/3)$ — expensive but affordable — so the
margin came from IPT's low iteration count and topped out around 4–123×. The
previous section concluded the advantage "was never no-factorization, it was
iteration count". That conclusion was right about the dense case and
**understated the sparse one**, where the two effects compound.

On sparse input, targeting an *interior* eigenvalue forces any Krylov method
into shift-invert, i.e. factorizing $(A - \sigma I)$. For sparsity with no
good elimination ordering — a random graph, as opposed to a lattice or a
banded matrix — the fill-in explodes:

| $N$ | nnz | LU fill-in |
|---|---|---|
| 2,000 | 33,880 | **88.7×** nnz |
| 5,000 | 84,868 | **222.1×** nnz |
| 10,000 | 169,866 | **430.7×** nnz |
| 20,000 | 339,826 | factorization no longer affordable |

IPT needs no factorization at all — 3–5 iterations, each one sparse matvec.

## Measured, against BOTH the standard tool and a fair matvec-only competitor

`bench_sparse.py`. The honest competitor is not only ARPACK: LOBPCG on
$(A-\sigma I)^2$ targets the interior with **no factorization either**, at the
cost of squaring the conditioning. Both are reported.

| $N$ | IPT | ARPACK shift-invert | LOBPCG $(A-\sigma)^2$ | **IPT vs best** |
|---|---|---|---|---|
| 2,000 | 0.0025 s | 1.00 s | 0.021 s | **8×** |
| 5,000 | 0.0045 s | 16.1 s | 0.104 s | **23×** |
| 10,000 | 0.0070 s | 140.6 s | 1.91 s | **273×** |
| 20,000 | 0.0179 s | *LU infeasible* | 6.19 s | **347×** |

Against ARPACK alone the ratio reaches ~20,000× at $N=10{,}000$. Against the
*best* alternative it is 8× → 347×, and **the margin grows with $N$**: IPT's
iteration count is set by its rate, not the matrix size (3–5 throughout),
while LOBPCG's squared-conditioning penalty worsens and the factorization
route dies outright.

## Verified correct, not merely low-residual

A solver that collapsed all four columns onto a single eigenvalue would also
show a small residual, so correctness is checked against dense ground truth
at sizes where that is affordable:

| $N$ | max $|\lambda - \lambda_{\mathrm{true}}|$ | relative to $\|A\|_2$ | distinct eigenvalues |
|---|---|---|---|
| 1,000 | 4.5e-13 | 4.6e-16 | 4 of 4 |
| 2,000 | 4.5e-13 | 2.3e-16 | 4 of 4 |

## It scales linearly, and the nonsymmetric case is larger still

Pushing $N$ up, symmetric, $k=4$ interior:

| $N$ | nnz | IPT | iters | rel. residual | time/nnz |
|---|---|---|---|---|---|
| 20,000 | 339,826 | 0.020 s | 3 | 2.9e-16 | 57 ns |
| 50,000 | 849,888 | 0.099 s | 2 | 7.6e-15 | 116 ns |
| 100,000 | 1,699,824 | 0.123 s | 3 | 7.3e-17 | 72 ns |
| 200,000 | 3,399,838 | **0.239 s** | 2 | 1.8e-15 | 70 ns |

Time per nonzero is flat (57–116 ns), i.e. genuinely $O(\mathrm{nnz})$, and
the iteration count does not grow with $N$. A 200,000-square problem with
3.4M nonzeros resolves four interior eigenpairs in a quarter of a second;
the shift-invert route needs an LU whose fill-in was already 431× nnz at
$N=10{,}000$.

**Nonsymmetric is the bigger win**, because the alternatives are thinner —
there is no LOBPCG for nonsymmetric problems, so `scipy.sparse.linalg.eigs`
with shift-invert is essentially the only option, and its fill-in is *worse*:

| $N$ | IPT | iters | ARPACK `eigs` shift-invert | LU fill | **speedup** |
|---|---|---|---|---|---|
| 2,000 | 0.0020 s | 4 | 0.78 s | 116.6× | **382×** |
| 5,000 | 0.0032 s | 3 | 12.1 s | 286.9× | **3,775×** |
| 10,000 | 0.0079 s | 3 | 105.2 s | 568.3× | **13,234×** |
| 50,000 | 0.043 s | 3 | *infeasible* | — | — |

Verified against dense ground truth at $N=800/1500$: max eigenvalue error
$1.4$–$4.7\times10^{-15}$ relative to $\|A\|_2$, four distinct eigenvalues,
and the eigenvectors come back correctly **non-orthogonal** (a symmetry
assumption leaking in would show as orthogonal output).

## Scope, stated honestly

This is the family IPT was designed for and it should be read that way: wide
diagonal spread, weak sparse coupling, so the per-column rate is tiny and the
basin condition is comfortably satisfied. It is a realistic family —
configuration-interaction Hamiltonians and random-network models look like
this — but it is *not* evidence about lattices, where GENERAL.md records that
2D Anderson defeats every method here including this one, nor about matrices
whose diagonal carries no spectral information. The `ipt_rate_columns` screen
(O(Nk), free) is what tells the two apart before committing.

Two questions decide whether the win applies to a given problem: *how many*
eigenpairs it survives, and *how strong* the coupling can get. Both measured,
reproducibly, with `python bench_sparse.py --envelope`.

**How many eigenpairs — no practical limit.** $N=20{,}000$ sparse symmetric,
interior targets:

| $k$ | time | iters | rel. residual | per eigenpair |
|---|---|---|---|---|
| 4 | 0.025 s | 3 | 9.2e-17 | 6.17 ms |
| 32 | 0.089 s | 3 | 2.4e-16 | 2.77 ms |
| 256 | 1.360 s | 3 | 6.4e-15 | 5.31 ms |
| 1,024 | 6.098 s | 4 | 3.9e-16 | 5.96 ms |

Cost is linear in $k$ at a flat ~6 ms/eigenpair, and the iteration count
stays at 3–4 across a 256× range of $k$ — the columns are independent, so
asking for more of them buys no coupling and no extra iterations. This is
therefore *not* a handful-of-eigenpairs method: a thousand interior
eigenpairs of a 20,000-square sparse matrix takes six seconds. (The
competitor table above was run at $k=4$; nothing here claims 347× still holds
at $k=1024$, only that IPT itself does not degrade.)

**How strong the coupling — and the finding that the threshold is not a
number.** Sweeping the off-diagonal scale on two instances:

| $N$ | coupling | $\rho_{\max}$ | outcome | iters | err |
|---|---|---|---|---|---|
| 5,000 | 5 | 0.0076 | converged | 5 | 7.6e-11 |
| 5,000 | 20 | 0.0304 | converged | 8 | 1.4e-11 |
| 5,000 | 40 | 0.0609 | converged | 10 | 1.7e-10 |
| 5,000 | 80 | **0.1217** | **converged** | 20 | 2.5e-10 |
| 5,000 | 160 | 0.2434 | diverges | 11 | 6.8e+02 |
| 2,000 | 20 | 0.0239 | converged | 12 | 3.5e-11 |
| 2,000 | 40 | 0.0478 | converged | 23 | 1.0e-10 |
| 2,000 | 80 | **0.0956** | **diverges** | 763 | 9.6e+01 |
| 2,000 | 160 | 0.1913 | diverges | 14 | 2.0e+02 |

The two instances **cross**: $N=5000$ converges at $\rho = 0.122$ while
$N=2000$ diverges at $\rho = 0.096$. So there is no universal $\rho$ at which
the method stops working — the screen is a one-hop quantity and the real
failure is driven by multi-hop resonances it cannot see, exactly the caveat
`ipt_rate_columns` already carries. What the sweep does establish is an
*empirically safe* region and a *definitely unsafe* one: $\rho \lesssim 0.05$
converged on every instance tried, $\rho \gtrsim 0.25$ diverged on every one,
and in between it is instance-dependent. The existing gate of 0.1 sits inside
the ambiguous band, which is the right place for it only because the
dispatcher treats a wrong screen as a cost, not a correctness risk.

Two operational consequences, both of which cost real debugging time here:

* **Cost degrades well before correctness does.** Iterations run 3–6 at
  $\rho \lesssim 0.02$ but 20–23 by $\rho \approx 0.05$–$0.12$ — a 4–8×
  slowdown while still returning machine-precision answers. The headline
  8–347× margins live in the low-$\rho$ regime; near the edge the method
  still *works* but much of the advantage is gone, surviving only where the
  alternative factorization has become infeasible outright.
* **Divergence can be slow, so `max_iter` can lie.** The $N=2000$,
  $\rho = 0.096$ row originally took **763 iterations** to declare failure,
  and a neighbouring case needed 76 iterations to *succeed*. With a tight
  `max_iter` the two are indistinguishable: an early run of this sweep
  reported divergence at $\rho = 0.12$ that was merely slow convergence under
  `max_iter=60`. This is now largely fixed (see below, 763 → 40), but the
  caveat remains: read `converged=False` as "did not converge in the budget
  given", and re-run with a larger budget before concluding the target is
  outside the basin.

## Two consequences of column-separability that were not being exploited

Both of the problems above are about *reporting*, and both had the same root
cause: the solver treated convergence as one fact about the batch when the
map makes it a fact about each column.

**A diverging target used to spoil its neighbours.** The abort was a global
test on $\max_j \|\Delta V_j\|$, so one bad column stopped the whole
iteration. Measured on a batch of four ($N=4000$, three isolated targets plus
one deliberately resonant one): the run aborted at iteration 6 and the *good*
columns came back at residual 4.4e-10, when alone they reach 5.7e-17 in 11
iterations. Retiring columns individually fixes this — the same batch now
returns 6.5e-16 / 2.0e-16 / 1.7e-15 on the three good columns and flags only
the fourth. Nothing is lost by retiring late, either: a converged column
already sits at its fixed point, and the columns never interact.

**The screen cannot say *which* column will fail — only the outcome can.**
This is the sharpest version of the "$\rho$ is a one-hop heuristic" caveat.
In a measured 4-target batch ($N=2000$, coupling 80):

| target | $\rho_j$ (screen) | outcome | residual |
|---|---|---|---|
| 0 | 0.064 | converged | 5.2e-15 |
| 1 | **0.096** (worst) | converged | 4.5e-15 |
| 2 | 0.086 | converged | 1.7e-15 |
| 3 | **0.042** (best) | **diverged** | 5.6e-08 |

The screen ranks the only failing target as the *safest* of the four. So
$\rho_j$ is not merely optimistic, it is **not monotonic**, and no gate value
could have routed this batch correctly. What makes the dispatcher sound is
therefore not the screen but the per-column outcome: `eig_partial` now falls
back on exactly the columns that failed, where before one failure sent all
four targets to ARPACK.

`ipt_eig_partial` reports `converged_cols`, `err_cols`, `iters_cols` and
`failed` alongside the scalar `converged` (which is now simply
`converged_cols.all()`, so existing callers are unaffected).

**Detecting divergence promptly.** A blow-up test ($\|\Delta V\| > 10^3\times$
its initial value) is a poor detector for a contraction factor just above 1:
the error creeps rather than explodes, which is what made that row take 763
iterations. Replacing it with a windowed *no-net-progress* test — no
improvement against the best error seen over the last `patience` iterations —
cuts that to **40**, with the other divergent rows falling 264 → 72 and
146 → 66. The test is windowed rather than per-step deliberately: a slowly
diverging column's step ratio dips below 1 often enough to keep resetting a
consecutive-step counter (that version still needed 420 iterations). Verified
not to cut short genuine slow convergence — cases needing 25, 40, 76 and even
322 iterations all still converge, and identically at `patience` 12 and 30.

**Cost of all this: about 3–7%, at the noise floor.** The per-column path is
not free, and two versions of it were measurably worse before this one. The
per-column maximum is the expensive part — reducing a C-ordered $(n,k)$ array
along axis 0 is a strided pass, measured 775 µs against 67 µs for the flat
maximum at $N=20{,}000$, $k=4$ — so the common iteration takes only the flat
maximum and per-column status is computed just when it can change something
(everything converged, something blew up, or once per stall window). Two
other regressions came from materializing an output buffer unconditionally
(an extra $(n,k)$ allocation plus a strided gather/scatter: 38% at $k=256$)
and from holding the `abs` temporary alive across iterations, which blocks
buffer reuse (~10% at small $k$). Interleaved A/B against the previous
implementation now reads 3.8 vs 3.5 ms ($N$=5000, $k$=4), 13.5 vs 13.1 ms
($N$=20,000, $k$=4) and 506 vs 492 ms ($N$=20,000, $k$=256), with one of
three rounds showing the new code faster on all three.
