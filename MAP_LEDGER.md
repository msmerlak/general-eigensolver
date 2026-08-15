# Map ledger

One line per mapping assessed, so a new exploration can see the whole field in
one read instead of re-deriving it from GENERAL.md. **Append here before
anything else.** Full write-ups stay in GENERAL.md; this is the index.

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
regimes the ledger records as unsolved (2D Anderson lattice; dense
far-from-diagonal below `sdc_eigvals` cost; exact degeneracy in a locator).

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
