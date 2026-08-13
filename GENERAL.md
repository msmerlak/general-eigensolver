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

The direction the literature points to, and the one this repository has *not*
tried, is non-orthogonal norm-reducing transformations — Eberlein-type shears,
which reduce the departure from normality rather than any off-diagonal norm.
That changes the invariant being decreased, which is precisely what failures 1
and 2 say is required. Untested here.
