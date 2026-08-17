"""Spectral divide and conquer for the general eigenproblem, via the matrix
sign function.

Motivation -- the measured inefficiency of the incumbent. On this CPU stack at
N=1000, with one gemm as the unit:

    gemm 1.00 | inverse 3.90 | QR 8.45 | pivoted QR 11.04 | dgeev 82.7

(That table was RE-MEASURED in SSJ_LOG #41 and the campaign's previous one --
inverse 5.35, QR 7.74, dgeev 93.9 -- was stale. The inverse is 27% cheaper
relative to gemm than every cost model here assumed, which is why two
inverse-for-gemm trades that looked like 13% wins measured as noise.)

A nonsymmetric eigendecomposition is only ~25 N^3 flops, i.e. ~12 gemms' worth
of arithmetic, yet dgeev costs 94 -- Hessenberg reduction and the QR iteration
are bandwidth- and latency-bound and run at roughly an eighth of gemm
efficiency. So a method that performs several times more *arithmetic* still
wins, provided the arithmetic is gemms. That is the opening this module aims
at, and it is a different bet from the rest of the repository: SSJ, IPT and the
shears all descend some invariant by small local updates, and GENERAL.md shows
each of them either stalls or has a bounded basin on non-normal input. Spectral
divide and conquer has no basin at all.

The method (Bai-Demmel-Gu style; the family is classical, nothing here claims
novelty):

  1. sign(A - sigma I) splits the spectrum by which side of the vertical line
     Re(z) = sigma each eigenvalue lies on. P = (I + S)/2 is the spectral
     projector onto the right half, and r = trace(P) is its rank.
  2. An orthonormal basis Q for range(P) (randomized range-finder, one
     unpivoted QR -- see _range_finder) block-triangularizes:
     Q^T A Q = [[A11, A12], [~0, A22]], with spec(A11) and spec(A22) the two
     halves. This is an ORTHOGONAL similarity, so it is unconditionally
     stable -- unlike the shears of normal.py, nothing here needs a
     well-conditioned eigenbasis.
  3. Recurse on A11 and A22 until blocks are 1x1, or 2x2 holding a complex
     conjugate pair.

The sign iteration itself reuses the repository's own pattern -- an expensive
globally convergent step while far away, a cheap gemm-only step once close:

    scaled Newton   X <- (mu X + mu^-1 X^-1)/2,  mu = |det X|^(-1/n)
    Newton-Schulz   X <- X(3I - X^2)/2                     (2 gemms, no inverse)

Newton is used while ||X^2 - I|| is large (it converges from anywhere with no
eigenvalues on the splitting line) and Newton-Schulz takes over once inside its
region, exactly as QR hands off to Newton-Schulz in ssj.core. Each avoided
inverse saves 3.90 gemm-equivalents and costs 2 -- a thinner margin than the
5.35 this module used to quote, and thin enough that two separate
inverse-for-gemm trades measured as noise (SSJ_LOG #41).
"""
from __future__ import annotations

import numpy as np

__all__ = ["sdc_eigvals", "matrix_sign"]


def matrix_sign(A, tol=1e-12, max_iter=60, ns_frob=1.0, count=None):
    """Matrix sign function by scaled Newton with a Newton-Schulz endgame.

    Returns (S, iters). `count` optionally accumulates a cost model in
    gemm-equivalents (inverse = 3.90, gemm = 1.0, re-measured in SSJ_LOG #41)
    so the benchmark can report where the time goes.

    Requires A to have no eigenvalues on the imaginary axis; the caller
    guarantees that by choosing (and if necessary perturbing) the shift. When
    one sits close enough that the iteration cannot converge within max_iter,
    this raises LinAlgError rather than returning an unconverged S -- see the
    note at the end of the loop.

    ns_frob : bound on ||I - X^2||_F for handing off to Newton-Schulz. 1.0 is
        GUARANTEED safe (||M||_2 <= ||M||_F, and NS converges inside
        ||I - X^2||_2 < 1) and is the default. Note this is a bound on the
        UN-normalized Frobenius norm, so in the loop's dev = ||.||_F/sqrt(n)
        units the threshold is 1/sqrt(n) -- a scaling law, not a constant.
        A fixed threshold on dev tests the wrong norm and breaks on spectra
        whose RMS deviation sits far below the operator norm: at dev < 0.9 a
        symmetric n=400 matrix enters NS outside its convergence region and
        never converges (SSJ_LOG #31).
    """
    n = A.shape[0]
    eye = np.eye(n)
    X = np.asarray(A, dtype=np.float64).copy()
    # Scale into range: the iteration is invariant to positive scaling, and
    # starting near unit norm keeps the first Newton steps well conditioned.
    # The scaling has to be BOTH cheap and tight. np.linalg.norm(X, 2) is an
    # SVD and costs more than several Newton steps; but the cheap bound
    # sqrt(||X||_1 ||X||_inf) overestimates a random matrix badly, which
    # over-scales X, pushes its eigenvalues toward 0 and *adds* Newton steps
    # (measured: 0.23x -> 0.08x of dgeev). Power iteration is both: O(N^2)
    # per step and tight within a few percent.
    nrm = _spectral_norm(X)
    if nrm == 0.0:
        return X, 0
    X /= nrm

    # THE FAR-FIELD GEMM IS OPTIONAL. A far-field Newton step costs X @ X
    # (2n^3) plus the LU and inverse (2n^3), and half of it is a gemm whose
    # only job is to evaluate ||X^2 - I||. But Newton already builds the whole
    # update, and
    #
    #     Delta = (X^-1 - X)/2 = X^-1 (I - X^2)/2,
    #
    # so the update norm is a convergence signal costing O(n^2) instead.
    #
    # Measured (Ginibre and near-symmetric, n=400 and 800): Delta tracks dev/2
    # to two digits through the endgame, and Delta is never small while dev is
    # large -- so gating on `delta_prev < ns_switch` reproduces the SAME
    # Newton-Schulz entry point, with the same iteration counts, while forming
    # X^2 once or twice instead of 8-11 times. X^2 is still built whenever the
    # gate opens, because the NS step consumes it and the handoff wants the
    # exact value rather than the proxy. `since_check` forces a real test every
    # 8 iterations so a pathological Delta sequence cannot ride to max_iter
    # without ever looking. (SSJ_LOG #30.)
    delta_prev = np.inf
    since_check = 0
    # ns_frob bounds ||I - X^2||_F; the loop carries dev = ||.||_F/sqrt(n).
    ns_switch = ns_frob / np.sqrt(n)

    for it in range(1, max_iter + 1):
        if delta_prev < ns_switch or since_check >= 8:
            since_check = 0
            X2 = X @ X
            if count is not None:
                count["gemm"] += count["_w"]
            dev = np.linalg.norm(X2 - eye, "fro") / np.sqrt(n)
            if dev < tol:
                return X, it
            if dev < ns_switch:
                # Newton-Schulz: 2 gemms, no factorization, quadratic.
                X = X @ (1.5 * eye - 0.5 * X2)
                if count is not None:
                    count["gemm"] += count["_w"]
                delta_prev = 0.0        # stay in the endgame: NS needs X^2
                continue
        else:
            since_check += 1

        # Scaled Newton with determinantal scaling, both through numpy's
        # LAPACK -- see _inv_and_logdet for why sharing one LU via scipy was a
        # 1.57x-2.07x LOSS rather than the saving it looked like.
        Xi, logabsdet = _inv_and_logdet(X)
        if not np.isfinite(logabsdet):
            raise np.linalg.LinAlgError("singular iterate in sign function")
        mu = np.exp(-logabsdet / n)
        if count is not None:
            count["inv"] += count["_w"]
        Xnew = 0.5 * (mu * X + Xi / mu)
        delta_prev = np.linalg.norm(Xnew - X, "fro") / np.sqrt(n)
        X = Xnew
        if not np.isfinite(delta_prev):
            raise np.linalg.LinAlgError("divergent iterate in sign function")

    # Running out of iterations is a FAILURE and must say so. It used to
    # return (X, max_iter), which is indistinguishable from succeeding on the
    # last allowed iteration, so a non-converged S propagated silently -- and
    # it does happen: a near-symmetric n=800 matrix exits here with
    # ||S^2 - I||/sqrt(n) = 3.4e-01 and a spectral count wrong by three
    # (SSJ_LOG #30). `sdc_eigvals` already catches LinAlgError and retries
    # with a different shift, which is exactly the right response, so raising
    # routes this into the existing recovery path instead of downstream.
    raise np.linalg.LinAlgError(
        f"sign iteration did not converge in {max_iter} iterations "
        f"(last update norm {delta_prev:.2e}); the shift is probably too "
        f"close to an eigenvalue")


def _spectral_norm(X, iters=20):
    """Tight O(N^2) estimate of ||X||_2 by power iteration on X^T X."""
    n = X.shape[0]
    v = np.ones(n) + 0.01 * np.arange(n)
    v /= np.linalg.norm(v)
    est = 0.0
    for _ in range(iters):
        w = X.T @ (X @ v)
        est = float(np.linalg.norm(w))
        if est == 0.0:
            return 0.0
        v = w / est
    return float(np.sqrt(est))


def _inv_and_logdet(X):
    """Inverse and log|det|, both through NUMPY's LAPACK.

    THIS USED TO CALL scipy.linalg.lu_factor / lu_solve, AND THAT WAS THE
    SINGLE LARGEST COST IN THE MODULE (SSJ_LOG #41). numpy and scipy ship
    SEPARATE OpenBLAS shared objects, each with its own thread pool. On a
    4-core box, alternating numpy gemms with scipy factorizations every
    iteration keeps both pools hot and they fight: measured 3.48x at n=512 and
    2.05x at n=1024 in a controlled same-arithmetic test, and scipy's
    lu_factor went from 3.6 ms in isolation to 70 ms inside this solver --
    19x, same routine, same size. End to end the boundary crossing cost this
    module 1.57x-2.07x.

    Sharing one LU between the determinant and the inverse was the reason for
    the scipy route, and it buys less than it looks: lu_factor + lu_solve
    against a full identity is 2n^3/3 + 2n^3 = 2.67n^3, and slogdet + inv is
    2n^3/3 + 2n^3 = 2.67n^3. Identical. So the "two factorizations" objection
    was never a flop argument, and the BLAS penalty makes it a clear loss.

    A zero pivot gives a non-finite logabsdet, which the caller turns into a
    retry with a different shift; that path is preserved.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        sgn, logabsdet = np.linalg.slogdet(X)
    if sgn == 0:
        return X, np.inf              # singular: caller retries another shift
    try:
        Xi = np.linalg.inv(X)
    except np.linalg.LinAlgError:     # pragma: no cover - defensive
        return X, np.inf
    return Xi, float(logabsdet)


def _split(A, shift, tol=1e-12, count=None, gate=1e-11):
    """One spectral split at Re(z) = shift.

    Returns (A11, A22, r) or None when the split is degenerate (everything on
    one side), which tells the caller to try another shift.
    """
    n = A.shape[0]
    if count is not None:
        count["_w"] = (n / count["_N"]) ** 3   # price sub-blocks by size^3
    S, _ = matrix_sign(A - shift * np.eye(n), tol=tol, count=count)
    P = 0.5 * (np.eye(n) + S)
    r = int(np.rint(np.trace(P)))
    if r <= 0 or r >= n:
        return None

    # Orthonormal basis for range(P) by a RANDOMIZED RANGE-FINDER:
    # QR([P G1, (I-P) G2]) with ONE unpivoted QR. This replaced
    # scipy.linalg.qr(pivoting=True) for two measured reasons (SSJ_LOG #41):
    # the pivoted QR ran at 6.0% of gemm rate -- 11.04 gemm-equivalents, the
    # worst-performing kernel in the whole method, worse than dgeev itself --
    # and it was a scipy call inside a numpy gemm loop, paying the separate-
    # OpenBLAS penalty documented at _inv_and_logdet.
    #
    # Column ORDER is load-bearing: the first r columns must come from P and
    # the rest from I - P. SSJ_LOG #24 records building the basis from a
    # pivoted QR of [P, I-P] instead, whose column reordering destroys the
    # range separation and reported a bogus ||A21||/||A|| = 2.6e-01 on a
    # symmetric matrix. P is idempotent to `tol`, so this split is exact.
    Q = _range_finder(P, r)
    if count is not None:
        count["qr"] += count["_w"]
    B = Q.T @ (A @ Q)
    if count is not None:
        count["gemm"] += 2 * count["_w"]

    # The (2,1) block is zero in exact arithmetic; how far it misses is the
    # split's backward error and is worth surfacing rather than assuming.
    resid = np.linalg.norm(B[r:, :r], "fro") / max(np.linalg.norm(A, "fro"), 1e-300)
    return B[:r, :r], B[r:, r:], r, resid


_G_CACHE = {}


def _range_finder(P, r, seed=0x5D1):
    """Orthonormal basis whose FIRST r columns span range(P), by one unpivoted
    QR of [P G1, (I - P) G2]. Deterministic: G is cached per size."""
    n = P.shape[0]
    if n not in _G_CACHE:
        _G_CACHE[n] = np.random.default_rng(seed).standard_normal((n, n))
    G = _G_CACHE[n]
    Y = np.empty((n, n))
    Y[:, :r] = P @ G[:, :r]
    Y[:, r:] = G[:, r:] - P @ G[:, r:]
    return np.linalg.qr(Y)[0]


def _base_eigvals(A):
    """Eigenvalues of a 1x1 or 2x2 block, in closed form."""
    if A.shape[0] == 1:
        return np.array([complex(A[0, 0])])
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    tr, det = a + d, a * d - b * c
    disc = tr * tr / 4.0 - det
    root = np.sqrt(complex(disc))
    return np.array([tr / 2.0 + root, tr / 2.0 - root])


def sdc_eigvals(A, tol=1e-12, min_block=None, max_depth=64, count=None,
                gate=1e-11, _depth=0, _rng=None):
    """Eigenvalues of a general real matrix by spectral divide and conquer.

    Globally convergent and free of any near-diagonal or normality
    requirement: every transformation applied is an orthogonal similarity.

    min_block : recurse until blocks are this small; default max(2, 3n//5),
        i.e. ONE split, both halves to dgeev. TWO measured constraints pin
        this, and they bracket it from opposite sides.

        From below: recursing to 2x2 (the original default) measured 4x
        slower on Ginibre n=400 -- 21.1x dgeev against 5.3x -- and no more
        accurate. Each level's split costs a full-size sign iteration while a
        dense solve of the block is milliseconds, so deep recursion pays
        splits to avoid work that was never expensive (SSJ_LOG #17, #25).

        From above: n//2 is too SMALL, which is the non-obvious half. The
        centred split returns r = trace(P), which lands near n/2 but
        essentially never on it, so one half comes back a few rows too big
        and triggers a whole second sign iteration -- and sign is ~2/3 of the
        run. Measured on Ginibre: 412 ms -> 207 ms at n=200 and 1315 ms ->
        748 ms at n=400 moving the leaf from 0.5n to 0.6n, accuracy
        unchanged at 4.4e-14 / 5.8e-14 (SSJ_LOG #28). The C port shows the
        same effect at 1.10x-1.16x, and it is flat from 0.55n to 0.9n, so the
        constant is not delicate -- what matters is being strictly above n/2
        while staying well below n, so a genuinely LOPSIDED split still
        leaves a big block that recurses.

    `count` optionally accumulates {"gemm", "inv", "qr"} operation counts.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if min_block is None:
        min_block = max(2, 3 * n // 5)
    if _rng is None:
        _rng = np.random.default_rng(0)
    if count is not None and "_N" not in count:
        count["_N"], count["_w"] = n, 1.0
    if n <= min_block or _depth >= max_depth:
        if n <= 2:
            return _base_eigvals(A)
        return np.linalg.eigvals(A).astype(complex)

    # Shift candidates: the mean eigenvalue (= trace/n) splits a spectrum
    # centred anywhere, then progressively perturbed shifts break ties when
    # the spectrum is clustered or symmetric about the first choice.
    centre = float(np.trace(A)) / n
    spread = float(np.linalg.norm(A, "fro")) / np.sqrt(n)
    for attempt in range(12):
        if attempt == 0:
            shift = centre
        else:
            shift = centre + spread * float(_rng.standard_normal()) * 0.5 ** (
                attempt // 4)
        try:
            out = _split(A, shift, tol=tol, count=count, gate=gate)
        except np.linalg.LinAlgError:
            continue
        if out is None:
            continue
        A11, A22, r, resid = out
        if resid > gate:
            # The gate WAS 1e-6, which never fired: measured headroom against
            # what the split actually achieves ran 25892x at n=1024 and 17
            # million x at n=256 (SSJ_LOG #39). A genuinely bad split -- an
            # eigenvalue 1.83e-04 from the splitting line, where the sign
            # function is ill-conditioned -- sailed through at ||A21|| =
            # 3.86e-11 while other shifts on the same matrix reached 1.6e-13.
            # At 1e-11 it is rejected, two retries find a good shift, and the
            # error improves 99x. The retry loop was always the right
            # machinery; only the threshold was wrong.
            continue  # split too inaccurate to trust; try another shift
        left = sdc_eigvals(A11, tol, min_block, max_depth, count, gate,
                           _depth + 1, _rng)
        right = sdc_eigvals(A22, tol, min_block, max_depth, count, gate,
                            _depth + 1, _rng)
        return np.concatenate([left, right])

    # No usable split found (e.g. a genuine multiple eigenvalue filling the
    # block): fall back rather than loop or return something wrong.
    return np.linalg.eigvals(A).astype(complex)
