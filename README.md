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

## Usage

```python
import numpy as np
from ssj import ssj_eigh

A = np.random.randn(500, 500); A = (A + A.T) / 2

w, V = ssj_eigh(A)                      # eigenvalues ascending, eigenvectors in columns
w, V = ssj_eigh(A, method="gemm")       # factorization-free: every flop a gemm
w, V, info = ssj_eigh(A, return_info=True)   # info["sweeps"], info["history"], ...
```

Complex Hermitian matrices work with the same call (anti-Hermitian generator,
unitary retraction).

### Methods

| `method` | retraction | when to use |
|---|---|---|
| `"auto"` (default) | Householder QR, switching to adaptive-depth Newton–Schulz once ‖K‖_F < ½ | general use |
| `"qr"` | Householder QR every sweep | reference / strictest orthogonality path |
| `"gemm"` | spectral cap ‖K‖₂ ≤ 1 + adaptive Newton–Schulz | hardware where gemm far outruns factorizations; ~2× flops of `"auto"` at the same sweep count, all of them gemms |
| `"cholqr2"` | CholeskyQR2 (gemm + small triangular ops) | stack-dependent; measured *slower* than QR on our CPU BLAS — see BENCHMARKS.md |

## Repository layout

- `src/ssj/core.py` — the solver (~200 lines of NumPy)
- `validate.py` — reproduces the convergence battery and scaling table from RESULTS.md
- `experiments.py` — mechanism experiments: trajectory, monotonicity, and controlled divergence of the variants that remove a saturation
- `tests/test_ssj.py` — correctness tests (run with `pytest` or directly)
- `ALGORITHM.md` — algorithm specification + implementation notes
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
