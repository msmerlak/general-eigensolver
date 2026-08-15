# SSJ improvement log

A dedicated track, separate from `MAP_LEDGER.md`. That ledger hunts for *new
maps*; this one improves the map the repository already ships. **Append here
before anything else**, one line per attempt, negative results included.

## What SSJ is, and where the cost sits

```
X ← I
repeat
    B ← XᵀA X ;  d ← diag(B)
    K_ij ← ½·atan( 2B_ij / (d_j − d_i) )      (±π/4 at zero gap; antisymmetric)
    X ← orth( X(I + K) )
```

Per sweep: 2 gemms (form B) + 1 orthogonalization (QR ≈ 0.67 gemm-equivalents,
or an adaptive Newton–Schulz endgame) ≈ **2.67 gemm-equivalents**.

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
| CholeskyQR2 retraction | slower than QR on this CPU BLAS |

## Open targets, roughly by expected value

1. **Cut the sweep count** (13–20 today). This is the dominant term and the
   most valuable direction. Shifts, deflation of converged columns, a better
   first sweep, anything that reduces iterations without breaking the mechanism.
2. **Cheapen the retraction** (0.67 of the 2.67 per sweep). The Newton–Schulz
   endgame already does this late; doing it earlier or cheaper is open.
3. **Exploit structure** — banded/sparse input, where forming XᵀAX densely is
   wasteful.
4. **Deepen mixed precision** beyond the current 1.3–1.4×.
5. **A convergence proof.** None is known. Genuinely valuable and the one item
   here that is a theory contribution rather than an engineering one.

## Attempts

| # | attempt | verdict |
|---|---|---|
| — | *(none yet)* | |
