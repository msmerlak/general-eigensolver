"""Simultaneous Saturated Jacobi (SSJ) eigensolver.

Full symmetric / Hermitian eigendecomposition by applying every classical
Jacobi rotation angle at once through a single linearized step, followed by
reprojection onto the orthogonal (unitary) manifold:

    B = X^H A X,  d = diag(B)
    K_ij = 1/2 * arctan( 2|B_ij| / (d_j - d_i) ) * phase(B_ij)   (anti-Hermitian)
           saturating at +-pi/4 * phase(B_ij) when d_j = d_i
    X <- orth( X (I + K) )

Two saturations make the map self-stabilizing: the arctan bounds each pair
angle by pi/4, and the reprojection of I + K (K anti-Hermitian) rotates each
K-invariant plane by arctan(sigma_l) rather than sigma_l, saturating the
*composed* step. Parameter-free; no pivoting, shifts, or deflation logic.

IMPORTANT: the retraction is applied to the PRODUCT X(I+K), never as
X @ orth(I+K). With an exact retraction the two coincide, but with the
truncated Newton-Schulz of the gemm variant only the product form re-measures
X's accumulated orthogonality defect inside Y^H Y and corrects it each sweep;
the factor form carries the defect forward and the iteration converges to a
non-orthonormal basis (observed: apparent off(B) at 1e-13 with a true
orthogonality error of X plateaued near 0.34).

The implementation is backend-agnostic between NumPy and CuPy: pass a CuPy
array and every operation dispatches to CuPy, so the whole iteration runs on
the GPU (results come back as CuPy arrays). The "gemm" method is the natural
GPU choice -- its only building blocks are matmuls and elementwise maps. See
bench_gpu.py.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import lapack as _lapack
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

__all__ = ["ssj_eigh", "off_frobenius"]


def _am(A):
    """Array module (numpy or cupy) for A -- dispatch without importing cupy
    unless a cupy array actually shows up."""
    if type(A).__module__.partition(".")[0] == "cupy":  # pragma: no cover
        import cupy
        return cupy
    return np


def off_frobenius(B) -> float:
    """Frobenius norm of the off-diagonal part of B."""
    xp = _am(B)
    off = B - xp.diag(xp.diag(B))
    return float(xp.linalg.norm(off, ord="fro"))


def _spectral_norm_estimate(A, iters: int = 100) -> float:
    """Estimate ||A||_2 by power iteration on the Hermitian matrix A.

    Deterministic start vector; a power-iteration estimate can only come in at
    or below the true norm, which makes the effective tolerance tighter, never
    looser. Avoids the O(N^3) SVD that an exact 2-norm would cost.
    """
    xp = _am(A)
    n = A.shape[0]
    v = 1.0 + 0.01 * xp.arange(n)
    v /= xp.linalg.norm(v)
    est = 0.0
    for _ in range(iters):
        w = A @ v
        est = float(xp.linalg.norm(w))
        if est == 0.0:
            return 0.0
        v = w / est
    return est


def _angles(B: np.ndarray) -> np.ndarray:
    """Saturated Jacobi generator: K_ij = 1/2 arctan(2 B_ij / (d_j - d_i)).

    B must be exactly Hermitian (symmetrize before calling); then K is exactly
    anti-Hermitian by construction. Conventions at the singular points:
    zero gap -> +-pi/4 * phase(B_ij); B_ij = 0 -> 0.
    """
    xp = _am(B)
    n = B.shape[0]
    d = xp.real(xp.diag(B))
    absB = xp.abs(B)
    gap = d[None, :] - d[:, None]  # gap_ij = d_j - d_i, exactly antisymmetric
    # Built in place: this map is O(n^2) but ran at 9.3 gemm-equivalents at
    # n=400 -- a third of the whole sweep -- almost all of it temporaries.
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = absB * 2.0
        theta /= gap
        xp.arctan(theta, out=theta)
        theta *= 0.5
    # Off the diagonal, gap_ij = 0 exactly when d_i == d_j, so the whole tie
    # question is answered by an O(n log n) pass over the diagonal instead of
    # three O(n^2) ones. With no ties the only non-finite entries are ON the
    # diagonal (gap_ii = 0), which fill_diagonal clears regardless, so the
    # nan_to_num scan is unnecessary too.
    if bool((xp.diff(xp.sort(d)) == 0.0).any()):
        # 0/0 (B_ij = 0 at zero gap): no rotation.
        theta = xp.nan_to_num(theta, nan=0.0)
        # Zero gap with B_ij != 0: the angle saturates at +-pi/4, but the raw
        # formula gives +pi/4 on BOTH (i,j) and (j,i) (each sees a +0 gap),
        # which breaks anti-Hermiticity -- orient the tie by the triangle.
        tie = (gap == 0.0) & (absB > 0.0)
        one = xp.ones((n, n), dtype=theta.dtype)
        upper = xp.triu(one, 1) - xp.tril(one, -1)
        theta = xp.where(tie, (np.pi / 4.0) * upper, theta)
    if B.dtype.kind == "c":
        nonzero = absB > 0
        phase = xp.where(nonzero, B / xp.where(nonzero, absB, 1.0), 0.0)
        K = theta * phase  # real theta times complex phase: needs a new array
    else:
        theta *= xp.sign(B)
        K = theta
    xp.fill_diagonal(K, 0.0)
    return K


def _orth_qr(M: np.ndarray) -> np.ndarray:
    """Orthonormal factor of M via QR, sign-fixed so diag(R) > 0 (the branch
    continuously connected to the identity)."""
    xp = _am(M)
    Q, R = xp.linalg.qr(M)
    d = xp.diag(R).copy()
    if d.dtype.kind == "c":
        absd = xp.abs(d)
        ph = xp.where(absd > 0, d / xp.where(absd > 0, absd, 1.0), 1.0)
    else:
        ph = xp.sign(d)
        ph = xp.where(ph == 0, 1.0, ph)
    return Q * ph


def _orth_cholqr2(M: np.ndarray) -> np.ndarray:
    """CholeskyQR2: two rounds of G = M^H M, R = chol(G), M <- M R^{-1}.

    Numerically safe here because sigma(X(I+K)) stays within a small factor of 1
    (each singular value of I+K is sqrt(1 + sigma_l(K)^2) times X's near-unit
    singular values), so the Gram matrix is well-conditioned; two rounds give
    orthogonality at the level of one Householder QR. R^{-1} is formed with
    trtri and applied by gemm, so the cost is two gemms per round plus two
    small triangular routines. Whether this beats Householder QR is a property
    of the BLAS/LAPACK build (it favors stacks where gemm far outruns
    factorizations, e.g. GPUs); on the CPU stack used for BENCHMARKS.md it
    measured ~3x slower, so it is an option, not the default.
    """
    xp = _am(M)
    for _ in range(2):
        G = M.conj().T @ M
        R = xp.linalg.cholesky(G).conj().T  # upper triangular
        if xp is np and _HAVE_SCIPY:
            trtri = _lapack.ztrtri if np.iscomplexobj(R) else _lapack.dtrtri
            Rinv, info = trtri(R, lower=0)
            if info != 0:  # pragma: no cover
                raise np.linalg.LinAlgError("trtri failed in CholeskyQR2")
            M = M @ Rinv
        else:  # pragma: no cover
            # R is well-conditioned here (see docstring), so a general inverse
            # is numerically safe and keeps the operation gemm-shaped on GPU
            M = M @ xp.linalg.inv(R)
    return M


def _orth_ns_adaptive(Y: np.ndarray, target: float, max_iter: int = 60) -> tuple[np.ndarray, int]:
    """Newton-Schulz iterated until ||Y^H Y - I||_F < target.

    Returns (Y, gemm_count). Each iteration's Y^H Y doubles as the error
    monitor. Requires sigma(Y) < sqrt(3) on entry.
    """
    xp = _am(Y)
    n = Y.shape[0]
    eye = xp.eye(n, dtype=Y.dtype)
    gemms = 0
    prev = np.inf
    for _ in range(max_iter):
        G = Y.conj().T @ Y
        gemms += 1
        dev = float(xp.linalg.norm(G - eye, ord="fro"))
        if dev < target or dev > 0.9 * prev:  # done, or stagnated at roundoff
            break
        prev = dev
        Y = Y @ (1.5 * eye - 0.5 * G)
        gemms += 1
    return Y, gemms


def _power_iter_norm_antisym(K, v0=None, iters: int = 25):
    """Estimate ||K||_2 for anti-Hermitian K by power iteration on K^H K.

    Returns (estimate, v) so the dominant vector can warm-start the next
    sweep's estimate -- the generator changes slowly sweep-to-sweep, and a
    warm-started estimate needs only a few iterations (a cold 30-iteration
    estimate every sweep measured ~2s of matvecs over a full N=1000 solve).
    A modest underestimate is safe: it weakens the cap slightly, and the
    sqrt(3)/sqrt(2) Newton-Schulz headroom absorbs it.
    """
    xp = _am(K)
    n = K.shape[0]
    if v0 is None:
        v = 1.0 + 0.01 * xp.arange(n)
    else:
        v = v0
        iters = 8
    v = v / xp.linalg.norm(v)
    est = 0.0
    for _ in range(iters):
        w = K.conj().T @ (K @ v)
        est = float(xp.linalg.norm(w))
        if est == 0.0:
            return 0.0, None
        v = w / est
    return float(np.sqrt(est)), v


_BLOCK_NS_FLOOR = 0.1  # Newton-Schulz target floor, as a fraction of tol
# How much margin to leave when predicting convergence, as a multiple of tol.
# Measured across 1e6/1e9/1e12 on cold GOE and on warm starts: 1e9 is fastest
# on both counts. Too thin (1e6) and a warm start beginning at rel_off ~3e-6
# never tightens at all, costing it a whole sweep; too wide (1e12) and cold
# solves tighten from the first sweep, giving back the saving. Tightening one
# sweep early is cheap -- a few Newton-Schulz iterations -- while tightening
# one sweep late costs an entire extra sweep, so the optimum sits wide.
_NS_TIGHTEN_MARGIN = 1e9


def _ns_target(target: float, block_m: int, tol: float,
               rel_off: float, prev_rel: float) -> float:
    """Newton-Schulz orthogonality target, tightened near convergence.

    The plain targets (rel_off^2, or gemm_ns_factor * rel_off) are calibrated
    for the gradual quadratic tail of the unpreconditioned map. With a block
    pass the error can fall four to six orders in ONE sweep, so the target is
    no longer below the NEXT error and the leftover orthogonality defect
    survives into the answer. Measured on a 1e-9 cluster with method="gemm":
    eigenvalue error 2.9e-15 with the floor against 4.5e-13 without, and
    orthogonality 9.5e-15 against 4.8e-12 -- the 2-3 digits this guards.

    But the defect only has to be small on the LAST sweep. The product-form
    retraction re-measures X's accumulated defect inside Y^H Y every sweep, so
    a loose defect earlier is corrected later; demanding 0.1*tol from the first
    sweep, when rel_off is still O(1), just buys Newton-Schulz iterations
    nobody needs. So predict the next error from the contraction just observed
    and tighten only when convergence is in reach. Measured: this is what makes
    method="gemm" usable with a block pass at all (n=1200, 9219 -> 6470 ms).
    """
    if not block_m:
        return target
    factor = rel_off / prev_rel if prev_rel > 0.0 else 1.0
    if rel_off * factor <= tol * _NS_TIGHTEN_MARGIN:
        return min(target, _BLOCK_NS_FLOOR * tol)
    return target


def _block_pass(B, X, m: int, offset: int):
    """One exact block-Jacobi pass on sorted-contiguous diagonal blocks of B.

    Sorts diag(B), cuts it into blocks of size m, diagonalizes each block
    exactly with a small dense eigensolve, and applies the block-diagonal
    orthogonal factor to both X and B. Returns (B, X) in the permuted basis --
    the whole SSJ map is permutation-equivariant, so the permutation is
    carried forward rather than undone.

    Why it helps: diag(B) starts at spread ||A||/sqrt(n) and has to climb to
    the true spectral spread, which costs ~(1/2)log n sweeps. While the spread
    is small nearly every gap is comparable to every coupling, so a large
    fraction of pairs saturate at +-pi/4 at once (7697 of them at n=400) and
    the contraction rate sits at 0.99. An exact block solve hands the
    iteration ~sqrt(m) of that spread immediately.

    Monotone for ANY grouping: for block-diagonal orthogonal P, the (I,J)
    block of P^H B P is Q_I^H B_IJ Q_J, whose Frobenius norm equals
    ||B_IJ||_F. Off-block masses are preserved individually while the
    within-block off-diagonal mass is zeroed, so off(B) can only fall.

    The sorted diagonal is rolled cyclically by `offset` instead of being cut
    into a ragged head and tail, so every block has size exactly m and ONE
    batched eigh replaces a Python loop over n/m small ones -- the loop is
    latency-bound on GPU. Trailing indices past the last full block are left
    for the next pass, which has a different offset.
    """
    xp = _am(B)
    n = B.shape[0]
    nb = n // m
    if nb < 2:
        return B, X
    keep = nb * m

    p = xp.argsort(xp.real(xp.diag(B)))
    if offset:
        p = xp.roll(p, -int(offset))
    B = B[p][:, p]
    X = X[:, p]

    idx = xp.arange(nb)
    blocks = B[:keep, :keep].reshape(nb, m, nb, m)[idx, :, idx, :]
    blocks = (blocks + blocks.conj().transpose(0, 2, 1)) / 2.0
    Q = xp.linalg.eigh(blocks)[1]  # batched over the leading axis
    # Apply Q block-diagonally. Materializing the (keep, keep) block-diagonal
    # factor and using one dense gemm costs n/m times the flops -- 3.0
    # gemm-equivalents per pass at n=800, m=32 against 0.12 here.
    #
    # Two ways to avoid that, and the better one depends on the backend. NumPy
    # wants the loop: each slice is contiguous, so the gemms read and write the
    # target in place with no temporary at all, and nb is only ~25. CuPy wants
    # the batched form: nb separate kernel launches are latency-bound on a GPU,
    # which is worth more than the reshape/transpose copies it costs.
    if xp is np:
        for b in range(nb):
            s = slice(b * m, (b + 1) * m)
            Qb = Q[b]
            X[:, s] = X[:, s] @ Qb
            B[:, s] = B[:, s] @ Qb
            B[s, :] = Qb.conj().T @ B[s, :]
    else:  # pragma: no cover - GPU path
        Qh = Q.conj().transpose(0, 2, 1)

        def _rmul(M):
            r = M.shape[0]
            return ((M.reshape(r, nb, m).transpose(1, 0, 2) @ Q)
                    .transpose(1, 0, 2).reshape(r, keep))

        X[:, :keep] = _rmul(X[:, :keep])
        B[:, :keep] = _rmul(B[:, :keep])
        B[:keep, :] = (Qh @ B[:keep, :].reshape(nb, m, n)).reshape(keep, n)
    return (B + B.conj().T) / 2.0, X


def ssj_eigh(
    A: np.ndarray,
    tol: float = 1e-13,
    method: str = "auto",
    max_sweeps: int = 1000,
    ns_switch: float = 0.5,
    gemm_cap: float = 1.0,
    gemm_ns_factor: float = 0.05,
    return_info: bool = False,
    X0: np.ndarray | None = None,
    precision: str = "full",
    mixed_switch: float = 1e-4,
    prologue: int = 0,
    block_m: int = 0,
    block_passes: int = 2,
    block_until: float = 0.0,
):
    """Eigendecomposition of a real symmetric or complex Hermitian matrix by
    Simultaneous Saturated Jacobi sweeps.

    Parameters
    ----------
    A : (n, n) symmetric / Hermitian ndarray.
    tol : stop when ||offdiag(X^H A X)||_F <= tol * ||A||_2.
    method :
        "auto"    -- QR retraction, switching to a single Newton-Schulz step
                     once ||K||_F < ns_switch (default; the spec's iteration).
        "qr"      -- Householder QR retraction every sweep.
        "cholqr2" -- CholeskyQR2 retraction every sweep (gemm-dominated,
                     usually faster in wall time; same NS endgame as "auto").
        "gemm"    -- factorization-free: spectral cap ||K||_2 <= gemm_cap, then
                     Newton-Schulz iterated until
                     ||Y^H Y - I||_F < gemm_ns_factor * off(B)/||A||_2.
                     Every flop a gemm.
    max_sweeps : hard cap on sweep count.
    return_info : if True, also return a dict with keys "sweeps", "history"
        (off(B)/||A||_2 before each retraction and at exit), "converged",
        "gemms" (N^3-gemm count, "gemm" method only), "norm_A".
    X0 : optional (n, n) orthonormal warm start (e.g. the eigenbasis of a
        nearby matrix). It is orthonormalized on entry, so a slightly drifted
        basis is fine. A warm start within off(B)/||A||_2 ~ 1e-2 of
        diagonalizing lands directly in the quadratic tail (~3-5 sweeps).
    precision : "full" (default) or "mixed". Mixed runs the linear phase in
        float32/complex64 down to off(B)/||A||_2 < mixed_switch, then
        warm-starts the full-precision phase from the low-precision basis.
        Sound because the map is memoryless -- every sweep re-derives its
        angles from a fresh B, so low-precision sweeps cannot poison the
        final accuracy, only the warm start's quality; the float64 tail
        restores full accuracy in a few sweeps. On CPU the low phase runs at
        sgemm speed (~2x); on tensor-core GPUs the same split is worth far
        more.
    mixed_switch : hand-off tolerance for the low-precision phase. The
        default 1e-4 is conservatively above float32's resolution; the low
        phase is also capped at 100 sweeps so a hand-off set too close to
        float32's floor degrades to a slightly worse warm start instead of
        stalling.
    prologue : number of unshifted QR-algorithm steps (B <- RQ, X <- XQ) to
        run before the first sweep. Off-diagonals decay like |lambda_i /
        lambda_j|^k under these steps, so a few of them collapse the sweep
        count for graded / rapidly decaying spectra (measured: 45 -> 5 sweeps
        with prologue=3 on spectrum 2^-i) and do nothing for flat spectra
        like GOE. Each step costs about one sweep. Ignored when X0 is given.
    block_m : block size for the block-cluster preconditioner (SSJ-BC).
        0 (default) disables it. May also be a sequence giving a PER-SWEEP
        schedule (last entry repeats): big blocks early inject the diagonal
        spread the first sweeps are bottlenecked on, small blocks late do the
        adjacent decoupling that is all the endgame needs. [n//2, n//4, 32]
        measured GOE n=800 at 8 sweeps against fixed-32's 15. Each sweep first runs `block_passes` exact
        block-Jacobi passes over sorted-contiguous blocks of diag(B), which
        collapses the saturated-pair population the plain map has to grind
        through. This is a preconditioner AROUND the map, not a change to it:
        the generator, the arctan saturation and the symmetric pairing are
        untouched. Measured sweep counts (GOE, method="auto"): 20/24/29 at
        n=200/400/800 falling to 9/11/14 at block_m=32, 69 -> 25 on exact
        5-fold degeneracy and 33 -> 9 on a 1e-9 cluster. It costs a batched
        eigh and two gemms per pass, so a sweep is dearer -- the sweep ratio
        is not a wall-time ratio.
    block_passes : block passes per sweep when block_m > 0. Consecutive passes
        alternate the block boundaries by block_m // 2 so pairs split across a
        boundary are caught by the next pass.
    block_until : stop running block passes once off(B)/||A||_2 falls below
        this. Default 0.0 (never stop), which measured strictly best: gating
        at 1e-3 costs sweeps on exactly the spectra the preconditioner exists
        for (5-fold degeneracy 25 -> 38, 1e-9 cluster 9 -> 12) and buys
        nothing back, because the warm-start regression such a gate would
        protect against does not occur here -- a tight X0 measures 2 sweeps
        with and without block passes. Kept as a knob, not a default.

    Returns
    -------
    w : (n,) real eigenvalues, ascending.
    V : (n, n) orthonormal eigenvectors, V[:, k] for w[k].
    info : dict, only when return_info=True.
    """
    xp = _am(A)
    A = xp.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if A.dtype.kind != "c" and A.dtype != np.float32:
        A = A.astype(np.float64, copy=False)

    if precision == "mixed":
        low_dt = np.complex64 if A.dtype.kind == "c" else np.float32
        _, V_low, info_low = ssj_eigh(
            A.astype(low_dt), tol=max(mixed_switch, tol), method=method,
            max_sweeps=min(max_sweeps, 100), ns_switch=ns_switch,
            gemm_cap=gemm_cap, gemm_ns_factor=gemm_ns_factor,
            return_info=True,
            X0=None if X0 is None else xp.asarray(X0).astype(low_dt),
            prologue=prologue, block_m=block_m, block_passes=block_passes,
            block_until=block_until,
        )
        full_dt = np.complex128 if A.dtype.kind == "c" else np.float64
        # The full-precision phase warm-starts past the spread problem, so the
        # schedule's big head blocks would be fp64 eigensolves fired on a
        # near-converged iterate -- measured pure waste (same sweep count).
        # But the SMALL tail blocks stay: structure below fp32's ~1e-7
        # resolution (tight clusters, exact ties) is invisible to the low
        # phase and lands on the fp64 phase unresolved, where small blocks
        # are exactly what resolves it (clustered 1e-9: 2 sweeps with the
        # tail blocks vs 5 without; 5-fold degeneracy: 18 vs 32).
        block_full = block_m if np.ndim(block_m) == 0 else block_m[-1]
        w, V, info = ssj_eigh(
            A.astype(full_dt), tol=tol, method=method,
            max_sweeps=max_sweeps, ns_switch=ns_switch, gemm_cap=gemm_cap,
            gemm_ns_factor=gemm_ns_factor, return_info=True,
            X0=V_low.astype(full_dt), block_m=block_full,
            block_passes=block_passes, block_until=block_until,
        )
        if return_info:
            info["sweeps_low"] = info_low["sweeps"]
            info["gemms_low"] = info_low["gemms"]
            info["history_low"] = info_low["history"]
            info["sweeps_total"] = info_low["sweeps"] + info["sweeps"]
            return w, V, info
        return w, V
    elif precision != "full":
        raise ValueError(f"unknown precision {precision!r}")

    n = A.shape[0]
    # block_m may be an int or a per-sweep schedule (a sequence; its last
    # entry repeats). The anatomy of the pre-cliff phase (SSJ_LOG attempt #9)
    # is why a schedule exists: early sweeps are bottlenecked on diagonal
    # SPREAD, which big blocks inject at ~sqrt(m) per pass, while late sweeps
    # only need adjacent pairs decoupled, which small blocks do at a fraction
    # of the cost. Measured: [n/2, n/4, 32...] takes GOE n=800 from 15 sweeps
    # to 8. On spectra whose bottleneck is not spread (exact degeneracy) a
    # schedule buys nothing -- keep a scalar there.
    # Never let a "block" be the whole problem: at block_m >= n the pass would
    # just be a call to the dense eigensolver, which is not what SSJ is.
    if np.ndim(block_m) == 0:
        block_sched = None
        block_m = min(int(block_m), n // 2)
        block_active = bool(block_m)
    else:
        block_sched = tuple(min(int(m), n // 2) for m in block_m) or (0,)
        block_m = block_sched[0]
        block_active = any(block_sched)
    norm_A = _spectral_norm_estimate(A)
    if norm_A == 0.0:  # zero matrix
        w = xp.zeros(n)
        V = xp.eye(n)
        return (w, V, {"sweeps": 0, "history": [0.0], "converged": True,
                       "gemms": 0, "norm_A": 0.0}) if return_info else (w, V)

    if X0 is None:
        X = xp.eye(n, dtype=A.dtype)
        if prologue > 0:
            B = A
            for _ in range(prologue):
                Q, R = xp.linalg.qr(B)
                B = R @ Q
                B = (B + B.conj().T) / 2.0
                X = X @ Q
    else:
        X0 = xp.asarray(X0)
        if X0.shape != (n, n):
            raise ValueError("X0 must match the shape of A")
        X = _orth_qr(X0.astype(A.dtype, copy=False))
    eye = xp.eye(n, dtype=A.dtype)
    history: list[float] = []
    gemms = 0
    converged = False
    sweeps = 0
    pow_v = None  # warm-started power-iteration vector ("gemm" method)
    prev_rel = 0.0  # previous sweep's rel_off, for the Newton-Schulz target

    for _ in range(max_sweeps):
        B = X.conj().T @ (A @ X)
        B = (B + B.conj().T) / 2.0  # exact Hermitian symmetry for the angle map
        rel_off = off_frobenius(B) / norm_A
        history.append(rel_off)
        if rel_off <= tol:
            converged = True
            break

        if block_sched is not None:
            block_m = block_sched[min(sweeps, len(block_sched) - 1)]
        if block_m and rel_off > block_until:
            for r in range(block_passes):
                B, X = _block_pass(
                    B, X, block_m, 0 if (sweeps + r) % 2 == 0 else block_m // 2)
            rel_blk = off_frobenius(B) / norm_A
            if rel_blk <= tol:
                converged = True
                history.append(rel_blk)
                sweeps += 1
                break

        K = _angles(B)
        Y = X @ (eye + K)  # = X + X @ K

        if method == "qr":
            X = _orth_qr(Y)
        elif method in ("auto", "cholqr2"):
            if float(xp.linalg.norm(K, ord="fro")) < ns_switch:
                # Adaptive-depth Newton-Schulz endgame. A single fixed NS step
                # (as in the original spec) leaves an O(||K||^4) orthogonality
                # defect that the last sweep never corrects; on graded spectra
                # the final K is not small even at convergence (tiny gaps keep
                # angles alive while off(B) is already at tolerance), and the
                # defect surfaces as an O(1e-9) orthogonality error in the
                # output. The target here is the quadratic-tail scale
                # rel_off^2: a defect below the square of the current error
                # cannot disturb the error-squaring tail (a looser
                # 0.05 * rel_off target was measured to stall the tail for a
                # sweep and break monotonicity at the 1e-7 level). Each NS
                # iteration contracts the defect quadratically, so this costs
                # ~2 extra gemms per endgame sweep -- still well below a QR
                # factorization; the stagnation guard floors an unreachably
                # tight target at roundoff.
                X, _ = _orth_ns_adaptive(Y, target=_ns_target(
                    rel_off * rel_off, block_active, tol, rel_off, prev_rel))
            else:
                X = _orth_qr(Y) if method == "auto" else _orth_cholqr2(Y)
        elif method == "gemm":
            gemms += 3  # B (2 gemms) + X @ K (1 gemm)
            sigma, pow_v = _power_iter_norm_antisym(K, v0=pow_v)
            if sigma > gemm_cap:
                K = K * (gemm_cap / sigma)
                Y = X @ (eye + K)
                gemms += 1
            # NS pre-scaling: the polar factor is invariant under positive
            # scalar scaling, and sigma(I+K) spans [1, sqrt(1+sigma_c^2)]
            # exactly (K anti-Hermitian), so dividing Y by the geometric
            # mean (1+sigma_c^2)^(1/4) centers the singular values around 1
            # and saves Newton-Schulz iterations while sigma_c is large.
            sigma_c = min(sigma, gemm_cap)
            if sigma_c > 0.1:
                Y = Y / (1.0 + sigma_c * sigma_c) ** 0.25
            X, ns_gemms = _orth_ns_adaptive(Y, target=_ns_target(
                gemm_ns_factor * rel_off, block_active, tol, rel_off, prev_rel))
            gemms += ns_gemms
        else:
            raise ValueError(f"unknown method {method!r}")
        prev_rel = rel_off
        sweeps += 1
    else:
        B = X.conj().T @ (A @ X)
        B = (B + B.conj().T) / 2.0
        history.append(off_frobenius(B) / norm_A)

    w = xp.real(xp.diag(B))
    # cupy's argsort takes no `kind`; numpy keeps the stable sort for
    # reproducible ordering within degenerate clusters
    order = np.argsort(w, kind="stable") if xp is np else xp.argsort(w)
    w = w[order]
    V = X[:, order]

    if return_info:
        return w, V, {"sweeps": sweeps, "history": history, "converged": converged,
                      "gemms": gemms, "norm_A": norm_A}
    return w, V
