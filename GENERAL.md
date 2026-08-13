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
