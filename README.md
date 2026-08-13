# general-eigensolver

A Python/NumPy implementation of **Simultaneous Saturated Jacobi (SSJ)**: full
symmetric / Hermitian eigendecomposition by applying *every* classical Jacobi
rotation angle at once through a single linearized step, followed by
reprojection onto the orthogonal manifold. Parameter-free — no pivoting, no
shifts, no deflation or degeneracy-detection logic; matrix multiplication and
an elementwise arctangent are the whole map.

```
X ← I
repeat
    B ← XᵀA X ;  d ← diag(B)
    if ‖offdiag(B)‖_F ≤ ε‖A‖₂ : break
    K_ij ← ½·atan( 2B_ij / (d_j − d_i) )      (±π/4 at zero gap; antisymmetric)
    X ← orth( X(I + K) )                       (QR, or Newton–Schulz endgame)
return sort(diag(B)), columns of X
```

The map is self-stabilizing through two automatic saturations: the arctan
bounds each pair angle by π/4, and the reprojection of I + K rotates each
K-invariant plane by arctan(σ) rather than σ, saturating the *composed* step.
See [ALGORITHM.md](ALGORITHM.md) for the full specification and
[RESULTS.md](RESULTS.md) for the original (Julia) measurements this
implementation reproduces. Our measurements, including two refinements found
during reimplementation and several confirmed negative results, are in
[BENCHMARKS.md](BENCHMARKS.md).

The repository also implements **IPT** (Iterative Perturbation Theory), a
one-gemm-per-iteration fixed point for near-diagonal input, and the
**SSJ→IPT hybrid** that uses SSJ's global basin to feed IPT's cheap endgame.
Where LAPACK is beaten, at full accuracy:

| problem | input | vs LAPACK | measured |
|---|---|---|---|
| symmetric | near-diagonal ($\rho \lesssim 0.03$) | **1.4–1.9×** faster than `dsyevd` | [BENCHMARKS.md](BENCHMARKS.md) |
| **general** | near-diagonal ($\rho \lesssim 0.1$) | **4–12×** faster than `dgeev` | [GENERAL.md](GENERAL.md) |

Both wins are *dispatchable*: applicability is decided by
$\rho = \max|W_{ij}|/|d_i-d_j|$, which `ipt_rate` computes in $O(N^2)$ —
free next to a single gemm — so a caller can pick IPT or LAPACK correctly
every time. [GENERAL.md](GENERAL.md) also records what does **not** work for
nonsymmetric matrices, and why.

## Usage

```python
import numpy as np
from ssj import ssj_eigh, ipt_eigh, ssj_ipt_eigh

# near-diagonal symmetric input: a handful of gemms, beating dsyevd
w, V = ipt_eigh(A_near_diagonal)

# GENERAL (nonsymmetric) near-diagonal input: 4-12x faster than dgeev
w, V = ipt_eig(A_general)          # eigenvectors are NOT orthogonal

# is IPT applicable? O(N^2), free next to a gemm
from ssj.ipt import ipt_rate
use_ipt = ipt_rate(A) < 0.5

# any symmetric input: SSJ globalizes, IPT finishes cheaply
w, V = ssj_ipt_eigh(A)

A = np.random.randn(500, 500); A = (A + A.T) / 2

w, V = ssj_eigh(A)                      # eigenvalues ascending, eigenvectors in columns
w, V = ssj_eigh(A, method="gemm")       # factorization-free: every flop a gemm
w, V, info = ssj_eigh(A, return_info=True)   # info["sweeps"], info["history"], ...
```

Complex Hermitian matrices work with the same call (anti-Hermitian generator,
unitary retraction).

**GPU:** pass a CuPy array and the whole iteration runs on the device — the
implementation is backend-agnostic between NumPy and CuPy. `method="gemm"` is
the natural GPU choice (matmuls and elementwise maps only). `bench_gpu.py`
measures SSJ against cuSOLVER's `syevd` on your GPU; see BENCHMARKS.md for
what to expect and why the CPU verdict may reverse there.

### Methods and accelerations

| `method` | retraction | when to use |
|---|---|---|
| `"auto"` (default) | Householder QR, switching to adaptive-depth Newton–Schulz once ‖K‖_F < ½ | general use |
| `"qr"` | Householder QR every sweep | reference / strictest orthogonality path |
| `"gemm"` | spectral cap ‖K‖₂ ≤ 1 + adaptive Newton–Schulz | hardware where gemm far outruns factorizations; ~2× flops of `"auto"` at the same sweep count, all of them gemms |
| `"cholqr2"` | CholeskyQR2 (gemm + small triangular ops) | stack-dependent; measured *slower* than QR on our CPU BLAS — see BENCHMARKS.md |

- `precision="mixed"` runs the linear phase in float32 and hands off to
  float64 for the quadratic tail — full final accuracy, ~1.3–1.4× on CPU,
  much more on tensor-core GPUs. Safe because the map is memoryless.
- `prologue=k` runs k unshifted QR-algorithm steps before the first sweep —
  collapses the sweep count for graded/decaying spectra (measured 45 → 5
  with `prologue=3`), does nothing for flat spectra.
- `X0=` warm-starts from a nearby eigenbasis (tracking: 1–5 sweeps).

Measured dead ends (see BENCHMARKS.md): over-relaxation γ·K slows or
diverges for every γ > 1, and generator-space momentum slows for every β
tried — the map tolerates no memory and no extra aggressiveness.

## Repository layout

- `src/ssj/core.py` — the solver (~200 lines of NumPy)
- `validate.py` — reproduces the convergence battery and scaling table from RESULTS.md
- `experiments.py` — mechanism experiments: trajectory, monotonicity, and controlled divergence of the variants that remove a saturation
- `tests/test_ssj.py` — correctness tests (run with `pytest` or directly)
- `src/ssj/ipt.py` — IPT and the SSJ->IPT hybrid (symmetric and general)
- `bench_vs_lapack.py` — head-to-head against LAPACK
- `experiments_general.py` — the nonsymmetric explorations behind GENERAL.md
- `ALGORITHM.md` — algorithm specification + implementation notes
- `GENERAL.md` — the general (nonsymmetric) problem: one win, three failures
- `RESULTS.md` — original measured results (Julia reference implementation)
- `BENCHMARKS.md` — measurements of this implementation

## Requirements

NumPy. SciPy is optional (used only for the LAPACK `trtri` inside the
`"cholqr2"` option). Python ≥ 3.9.

## Honest context

This is not CPU-competitive with LAPACK from a cold start (`dsyevd` wins by
~50× at N=1000 on our box; see BENCHMARKS.md). The niches are (a) hardware
where an eigensolve costs many gemm-equivalents and the `"gemm"` variant's
pure-multiplication diet applies, and (b) robustness: a three-line,
parameter-free method with an empirically global basin and native degeneracy
handling. No convergence proof is known; see the "Not established" section of
RESULTS.md.
