# Map ledger — CLOSED

**The map hunt is closed at 34 entries.** This file is now a read-only record;
the active work is in `OPTIMIZATION_LOG.md`. Kept because the negative results are the
point: they cost real measurement and they make future work cheap.

## Why it was closed

Marginal value per iteration was clearly declining. The last four entries
(#31–#34) were all well-characterized losses, and the untouched ground left —
hierarchical/H-matrix, tensor structure — consists of large established fields
where a fresh attempt is unlikely to beat decades of specialized work.

More importantly, the campaign produced a *structural* answer rather than a
list of failures. **Four independent attempts to escape the divide-by-gap
generator each landed back on it**, one of them provably (ρ(J) is a
diagonal-similarity invariant, so no coordinate reconditioning can widen any
locator basin). The generator, the arctan saturation on it, and the symmetric
pairing that gives it descent are **one mechanism**, and the space of
plausible-looking fixes around it is essentially empty. That is a finding, and
it is why continuing to sample that space had stopped paying.

The wins that do exist all came from the same move: **find a regime where the
incumbent's structure fails**, not from outrunning it on its home ground.
Sparse interior eigenpairs win because LU fill-in explodes on unstructured
sparsity and IPT needs no factorization at all. Nothing here ever beat LAPACK
or ARPACK at what they are good at, and the lessons below explain why that was
predictable.

## What remains open, for anyone picking this up

* **GPU validation of SSJ — measured, and it retired the thesis on this card.**
  `bench_gpu.py` ran on a Tesla T4 (`OPTIMIZATION_LOG.md` #33): cuSOLVER's `syevd` is
  *more* gemm-efficient on GPU than on CPU, not less, closing the room the
  all-gemm premise needed. The verdict is card-specific (a T4's fp64 rate is
  unusually low, deflating every gemm-equivalent count including cuSOLVER's
  own — see #33's caveat), so an A100 re-run is still open, but the framing
  that no GPU measurement existed no longer holds. SDC's GPU story is
  different and better: a narrow, verified win over `cupy.linalg.eig` at
  n ≤ 512 (#36, #42).
* **A convergence proof for SSJ's linear phase**, now more tractable: descent
  held on every seed ever run, and the ≈½·log n diagonal-spread mechanism is
  quantified in `OPTIMIZATION_LOG.md`.
* **A fast Cauchy apply (FMM) for the secular map (#28)** — the only basin-free
  exact method here, O(n²) but losing on constants with crossover near n≈5000.
  Known technology, known endpoint.

---

One line per mapping assessed. Full write-ups stay in GENERAL.md; this is the
index.

Incumbents to beat, by regime:

| regime | incumbent | to beat |
|---|---|---|
| dense symmetric, cold start | LAPACK `dsyevd` | ~30× faster than SSJ today |
| global symmetric, matmul-only | `ssj_eigh` | 13–20 sweeps ≈ 35–53 gemm-equiv |
| symmetric near-diagonal | `ipt_eigh` | 1.4–1.9× `dsyevd` |
| general near-diagonal | `ipt_eig` | 4–12× `dgeev` |
| k interior eigenpairs, sparse | `ipt_eig_partial` | 8–347× best alternative |
| dense non-normal | `sdc_eigvals` | 0.13–0.22× `dgeev` |

## Assessed

| # | mapping | state / mechanism | verdict |
|---|---|---|---|
| 1 | IPT | vector, divide-by-gap | **shipped** — 1.4–1.9× `dsyevd`, 4–12× `dgeev` in basin |
| 2 | IPT w/ Rayleigh λ | vector | no gain over IPT |
| 3 | damped IPT | vector | no gain |
| 4 | Aitken/Shanks on IPT | sequence extrapolation | no gain |
| 5 | scalar self-energy | scalar | no gain |
| 6 | Richardson / Jacobi | vector | no gain |
| 7 | Anderson acceleration on IPT | history mixing | no basin extension at any depth |
| 8 | Davidson | subspace | **shipped** — 16× basin, faster inside it |
| 9 | Jacobi–Davidson (2 variants) | subspace + correction eq. | no improvement or strictly worse |
| 10 | block IPT (dense, adaptive) | block/invariant subspace | **shipped** — up to 100× basin |
| 11 | sparse block IPT | block, matvec-only | **shipped** — band ×4, 26–54× at N=5000 |
| 12 | Riccati–Newton (rank-one restored) | vector | modest gain only |
| 13 | Riccati–Chebyshev (3rd order) | vector | no better than Newton |
| 14 | Brillouin–Wigner (`bw_eig_partial`) | vector, self-consistent denom. | **shipped** — 70→106 of 240, same cost |
| 15 | `gipt` preconditioned inverse iter. | vector | **shipped** — ~60× basin |
| 16 | SSJ | subspace, isospectral rotation | **shipped** — global, parameter-free |
| 17 | LOBPCG for k eigenpairs | subspace | 10–100× slower than ARPACK |
| 18 | Chebyshev-filtered subspace iter. | polynomial filter | 0.6–0.75× at matched accuracy |
| 19 | Lanczos-seeded Davidson | subspace start | no gain |
| 20 | purification 3P²−2P³ | projector | **shipped** — `window_eig`, certified count |
| 21 | matrix sign / SDC | projector | **shipped** — `sdc_eigvals`, 0.13–0.22× `dgeev` |
| 22 | Zolotarev rational | rational filter | needs partial fractions; not competitive |
| 23 | Brockett double bracket | isospectral gradient flow | ~6,000× SSJ; ~800× with momentum |
| 24 | QR flow (unshifted) | factor-and-swap | 7–10× SSJ |
| 25 | Cholesky-LR flow | factor-and-swap | 7–10× SSJ; best of the flow losers |
| 26 | homotopy + basis refresh | path following | globally convergent; 26–248× SSJ; NaN on exact degeneracy |
| 27 | one-sided Jacobi → Schur form | triangular fixed point | no descent property; stalls 1/12; 68× `dgees` |
| 28 | secular equation (rank-one) | scalar root-finding | exact, **no basin at any coupling**, O(n²); crossover ~n=5000 |
| 29 | Oja / Rayleigh gradient flow | vector gradient | structurally wrong for interior targets |
| 30 | randomized range finder + IPT | sketch | fails — gives a subspace, IPT needs a frame |
| 31 | Perron balancing of the gap-weighted coupling graph | positive diagonal; fixed point is a Perron ray | **no-go, proved and verified** — 0 of 15 basin cases changed; ρ(J) is exactly invariant under diagonal similarity (8.5e-16), so no reconditioning can move the basin. By-product screen ρ(\|J\|) is a better classifier (AUC 0.988 vs 0.944) but costs more than the solve it screens |
| 32 | inertia-certified Laguerre on the log-det jet | banded LDL^T carried as a 2nd-order Taylor jet in σ; one pass gives ν (Sylvester) + tr R + tr R²; no gap denominator anywhere | **loses as an eigensolver** — 25–147× flops, 317–1149× wall vs ARPACK shift-invert, which amortizes ONE factorization over all k while this refactorizes at every shift. **Narrow win**: certified window count 2.0–14.0× over `window_count` (#20), crossover N≈250, banded-only, and the certificate is exact on generic shifts but wrong 23/200 with \|Δν\| up to **248** for shifts within 1e-13…1e-9 of an eigenvalue |
| 33 | stochastic moment measure (SLQ / KPM quadrature of the spectral measure) | a random *measure* carried as scalars; state is shift-independent; fixed point is the spectral measure itself | **loses** — 334× flops / 9,700× wall vs `eigvalsh`, 2,660× flops vs #32, on exact counting. Two floors multiply: resolution **bias** ≈ n/degree (probes cannot touch it) and Hutchinson variance ≈ 90r probes. Exactness needs degree ≳ n — the same order as diagonalizing. Shift-independence is real but amortizes a per-boundary cost that was already ~1e-2 gemm-equiv |
| 34 | commutant map (automorphism detection + isotypic block reduction) | a discrete **group element** π; fixed-point set is Aut(A), which carries no spectral information | **conditional win, measure-zero condition.** Beats `dsyevd` where an exact free cyclic automorphism exists, but the reduction is classical (Bloch / symmetry-adapted bases) and the condition has a hard cliff at relative defect ~1e-12 — τ swept 1e-13→1e-4 changed **0 of 8** decisions. Capped at ≈4n/W ≈ 57 by its own Θ(n²) detection, not by g²/2 = 512. Declines correctly on GOE, but costs 81–222% of the ARPACK solve to decline on 2D Anderson — **empty in the open regime**. Exact (5.4e-15) on symmetry-induced degeneracy where `ipt_eigh` returns NaN |

## "Unsolved" means unsolved BY THIS REPO, not unsolved

Stated precisely, because a loose reading of it would let a candidate clear the
breakthrough bar without beating anything real. In every regime below there is
already a method that works well; what is missing is a *matmul-only or
factorization-free* route to it.

| regime | who already solves it | what is actually open |
|---|---|---|
| 2D Anderson lattice | ARPACK shift-invert, comfortably — **3.9 solves per eigenpair** at the band centre (#32) | doing it factorization-free; every matvec-only method here fails |
| dense far-from-diagonal | LAPACK `dgeev`; `sdc_eigvals` reaches only 0.13–0.22× of it | beating `dgeev`, not beating `sdc_eigvals` |
| exact degeneracy | LAPACK and ARPACK handle it; SSJ handles it natively via the arctan saturation | locators (IPT family) fail on it — a locator that did not would be new |

So a breakthrough must beat **the best available method**, LAPACK and ARPACK
included — not merely the best method in this repository. Beating an incumbent
that is itself far off LAPACK is not a win; #33 beat `window_count` by 20.7× in
flops and that was worthless, because `window_count` is itself 6,900× LAPACK's
flops on the same task.

## Measurement hygiene on this container

Wall-clock numbers here are only meaningful on a **quiet** machine. Measured
during this campaign, with two autonomous agents benchmarking concurrently:
LAPACK `eigh` at n=128 timed at **10 s** against its true ~2 ms — a ~5000×
inflation — with load average 6.3 on 4 cores. An earlier concurrent run
inflated a LAPACK baseline 0.1 s → 9 s and produced a bogus 26–47× claim that
was nearly shipped.

So: **hardware-free units are primary** — sweeps, gemm-equivalents, matvecs,
modelled flops with the same constants on both sides. Those are immune, which
is why every entry above reports them. Before trusting any wall-clock ratio,
check `uptime` and re-measure sequentially with nothing else running.

## Recurring lesson

**Four** independent attempts to escape the divide-by-gap generator each landed
back on it: replacing the division (Brockett) costs ~800×, avoiding the need
for it (homotopy) costs 26–248× and dies on degeneracy, changing its target
(Schur) loses the descent property, and reconditioning its coordinates (Perron
balancing) is provably a no-op — ρ(J) is a diagonal-similarity invariant. The
generator, the arctan saturation on it, and the symmetric pairing that gives it
descent are **one mechanism**.

So a candidate that merely re-parameterizes the locator is very unlikely to
pay. The open ground is elsewhere: state types nobody has used here
(stochastic/sampling dynamics, moments or continued-fraction coefficients,
factorizations as state, hierarchical/low-rank off-diagonal structure), or the
regimes below where this repository's distinctive approach does not yet win.

A second lesson, from #31: **when the outcome is cheaper to observe than to
predict, invest in cheap failure, not better prediction.** A candidate whose
value is a better convergence *predictor* must beat the cost of simply
attempting the solve — measured, ρ(|J|) did not.

A third, from #32, and the sharpest filter yet on candidates: **a method that
itself needs factorizations can never win the sparse/interior regime**, because
shift-invert amortizes *one* factorization across all $k$ eigenpairs while any
shift-varying method pays a new one per shift (measured: ~1400
factorization-equivalents against 1). The entire reason `ipt_eig_partial` wins
there by 8–347× is that it needs *none*. So a candidate aimed at that regime
must be genuinely factorization-free; if it factorizes at all, it is competing
on ARPACK's home ground with a handicap.

A fourth, also from #32: **a completeness certificate protects the count, not
the value.** It can be perfectly self-consistent around a wrong answer. Do not
accept "the certificate agrees" as evidence of accuracy.

A fifth, from #33, and the exact **dual of the third**: an $LDL^\top$ buys an
exact integer count in $O(nb^2)$ with no dependence on the boundary gap, while
every matvec-only route pays $\Theta(n)$ polynomial degree for that same
integer — verified here independently, with the variance channel removed, the
count error falls only with degree and needs $\sim\!2n$–$4n$ to go below 0.5.
So **exact counting is where factorization is structurally unbeatable**, exactly
as being factorization-free is what wins for sparse interior eigenpairs. The
two lessons bracket the regime: match the tool to which of the two questions is
being asked, because no single mechanism wins both.

A sixth, from #34, about **where the cost actually goes once you beat n³**: the
O(n³/g²) block solve became **0.6% of the total call**, and with eigenvectors the
answer's own memory footprint — an n×n *complex* matrix — cost **16× the solve**.
So the speedup was capped by the method's own Θ(n²) detection at ≈4n/W ≈ 57,
not by the group at g²/2 = 512. **Any method that reduces n³ to n³/g² on a
dense problem runs into its own Θ(n²) floor**, and complex output doubles it.
Check that floor before assuming an asymptotic win survives contact.

A seventh, also from #34: a knob that cannot change any decision is not a
tolerance. Sweeping τ over nine orders of magnitude changed 0 of 8 accept/reject
outcomes, because the exact verifier sat downstream of an exact-match
constructor — the cliff was in construction, not in verification. If a
parameter never moves an outcome, the robustness it appears to offer is
imaginary.
