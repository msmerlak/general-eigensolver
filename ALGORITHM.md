# Simultaneous Saturated Jacobi (SSJ)

Full symmetric eigendecomposition by applying **every** classical Jacobi
rotation angle **at once**, through a single linearized step, followed by
reprojection onto the orthogonal manifold. Parameter-free; matrix
multiplication and elementwise arctangent only.

## Iteration

**Input:** symmetric $A \in \mathbb{R}^{N\times N}$, tolerance $\varepsilon$.
**State:** orthonormal $X \in O(N)$, initialized $X_0 = I$.

Repeat until converged:

**1. Ritz block.**

$$B = X^{\mathsf T} A X, \qquad d_i = B_{ii}$$

**2. All rotation angles, saturated.** Build antisymmetric $K$:

$$K_{ij} = \frac{1}{2}\arctan\!\left(\frac{2B_{ij}}{d_j - d_i}\right),
\qquad K_{ji} = -K_{ij}, \qquad K_{ii} = 0,$$

with the limit convention $K_{ij} = \frac{\pi}{4}\,\mathrm{sign}(B_{ij})$ when
$d_j = d_i$ (and $K_{ij}=0$ when $B_{ij}=0$). This is the exact Jacobi angle
$\tan 2\theta = 2B_{ij}/(d_j-d_i)$, principal branch $|\theta|\le\pi/4$; its
small-angle limit is the first-order correction $B_{ij}/(d_j-d_i)$.

**3. Linearized step + reprojection.**

$$X \leftarrow \mathrm{orth}\big(X\,(I + K)\big)$$

where $\mathrm{orth}$ is the QR factor — or, once $\|K\|_F < \tfrac12$, a
single Newton–Schulz step $Y\big(\tfrac{3I - Y^{\mathsf T}Y}{2}\big)$, which is
pure matrix multiplication.

**4. Stop** when $\|\mathrm{offdiag}(B)\|_F \le \varepsilon\,\|A\|_2$.
**Output:** eigenvalues $d_i$, eigenvectors = columns of $X$ (sort by $d_i$).

## Factorization-free variant (gemm-only)

Replace step 3 by: cap the step spectrally,
$K \leftarrow K \cdot \min(1,\, 1/\|K\|_2)$ ($\|K\|_2$ by power iteration),
then orthonormalize with Newton–Schulz iterated until
$\|Y^{\mathsf T}Y - I\| < 0.05\,\|\mathrm{offdiag}(B)\|_F/\|A\|_2$ — each
step's $Y^{\mathsf T}Y$ doubles as the error monitor. The cap keeps
$\sigma(I+K) \le \sqrt{2} < \sqrt{3}$, inside the Newton–Schulz convergence
region. Same sweep counts as QR, ~2× flops, every flop a gemm.

## Pseudocode

```
X ← I
repeat
    B ← XᵀA X ;  d ← diag(B)
    if ‖offdiag(B)‖_F ≤ ε‖A‖₂ : break
    K_ij ← ½·atan( 2B_ij / (d_j − d_i) )   for i < j   (±π/4 at zero gap)
    K ← antisym(K)
    X ← orth( X(I + K) )                    (QR, or NS when ‖K‖_F < ½)
return sort(diag(B)), columns of X
```

## Properties

- **Fixed points:** $K = 0 \iff B$ diagonal $\iff$ $X$ is an eigenbasis. The
  iteration never changes the problem's solutions.
- **Two saturations:** the arctan bounds each pair angle by $\pi/4$; and for
  antisymmetric $K$ the polar factor of $I+K$ rotates each $K$-invariant plane
  by $\arctan\sigma_\ell$ — the reprojection saturates the *composed* step.
  Both are automatic; neither is a tunable safeguard.
- **Degeneracies need no handling:** at gap $0$ the angle saturates and the
  pair resolves as under sequential Jacobi.
- **Endgame is quadratic** (small angles ⇒ Newton step); the global phase is
  linear, with sweep count growing $\approx O(\log N)$ empirically.
- **Cost:** ~5 gemm-equivalents per sweep (QR variant); $O(N^3\log N)$ cold.
- **Do not** "improve" the step toward the true rotation (e.g.
  $I+K+\tfrac{K^2}{2}$), extrapolate across sweeps, or defer the
  reprojection — each removes the second saturation and diverges.

---

## Implementation notes (this repository)

Two refinements over the spec as written above, both measured in
[BENCHMARKS.md](BENCHMARKS.md):

1. **The retraction must be applied to the product** $X(I+K)$, never as
   $X\cdot\mathrm{orth}(I+K)$. With an exact retraction the two coincide; with
   the truncated Newton–Schulz of the gemm variant only the product form
   re-measures $X$'s accumulated orthogonality defect inside $Y^{\mathsf T}Y$
   and corrects it every sweep. The factor form converges in apparent
   $\mathrm{off}(B)$ while $X$ drifts $O(1)$ from orthogonality.

2. **The endgame Newton–Schulz is adaptive-depth**, iterated until
   $\|Y^{\mathsf T}Y - I\|_F < 0.05\,\mathrm{off}(B)/\|A\|_2$ — the same rule
   the gemm variant already uses — rather than a single fixed step. A single
   step leaves an $O(\|K\|^4)$ defect that the last sweep never corrects, and
   on graded spectra the final $K$ is *not* small at convergence (tiny gaps
   keep angles alive while $\mathrm{off}(B)$ is already at tolerance): measured
   $8\times10^{-9}$ output orthogonality error, restored to roundoff by the
   adaptive rule at ~2 extra gemms per endgame sweep.

Plus one extension: the same map handles **complex Hermitian** matrices with
$K_{ij} = \frac12\arctan\!\big(2|B_{ij}|/(d_j-d_i)\big)\cdot B_{ij}/|B_{ij}|$
(anti-Hermitian $K$, unitary retraction); GUE converges with the same sweep
profile as GOE.

One vectorization trap: at an *exact* zero gap the full-matrix formula
$\frac12\arctan(2B_{ij}/(d_j-d_i))$ evaluates both $(i,j)$ and $(j,i)$ with a
$+0$ gap, producing $+\pi/4$ on **both** triangles — silently breaking the
antisymmetry of $K$. The tie case needs its orientation set explicitly by the
triangle ($+\pi/4$ above the diagonal, $-\pi/4$ below, times the phase).
