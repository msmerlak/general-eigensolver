"""Mapping #32 candidate: INERTIA-CERTIFIED LAGUERRE SLICING.

STATE          a real shift sigma together with the LDL^T factorization of
               A - sigma I, carried as a SECOND-ORDER TAYLOR SERIES in sigma.
               No vector, no subspace, no projector, no matrix iterate.

WHAT ONE PASS RETURNS.  Factor A - (sigma + eps) I = L(eps) D(eps) L(eps)^T
with every scalar carried as x0 + x1 eps + x2 eps^2.  Then, writing
l(sigma) = log|det(A - sigma I)| = sum_i log|sigma - lambda_i|,

    nu(sigma) = #{ j : d0_j < 0 }            = #{ i : lambda_i < sigma }   (Sylvester)
    s1(sigma) = sum_j d1_j / d0_j            = tr( (sigma I - A)^-1 )      = l'
    s2(sigma) = sum_j (d1_j^2/d0_j^2 - 2 d2_j/d0_j)
                                             = tr( (sigma I - A)^-2 )      = -l''

so ONE factorization pass yields an exact integer index and the first two
resolvent trace moments.  That is the piece that makes this cheap for a
general BANDED matrix; for a tridiagonal it is the classical Sturm-sequence
derivative, but for bandwidth b > 1 the usual route to tr(R) is a selected
inversion, which costs more than the factorization it post-processes.

THE MAP.  With a bracket [a,b] certified by inertia to contain exactly the
targets of index t, and m = nu(b) - nu(a) the multiplicity,

    sigma  <-  sigma  -  n / ( s1  +-  sqrt( (n/m - 1) (n s2 - s1^2) ) )

which is Laguerre's iteration on the characteristic polynomial.  For a
real-rooted polynomial Laguerre converges monotonically and cubically from
ANY point in a root-free interval and provably cannot leave it, so there is
no basin condition of any kind.  n s2 - s1^2 >= 0 by Cauchy-Schwarz, so the
square root is always real.

NO GAP APPEARS ANYWHERE.  Every fast map in the ledger divides by
(d_j - d_i); here the only denominators are s1 +- sqrt(...), whose size is set
by the distance to the TARGET, not by the distance between two eigenvalues.
Exact degeneracy is not a special case: it makes nu jump by m, and m enters
the formula as a known integer that restores cubic convergence.
"""
from __future__ import annotations

import numpy as np

__all__ = ["taylor_ldl_band", "to_band", "laguerre_eigval", "bisect_eigval",
           "eigvec_from_shift", "Count", "sweep_window"]


class Count:
    """Hardware-free cost.  `fact` counts Taylor-2 banded LDL^T passes;
    `flops` charges each pass its true arithmetic."""

    def __init__(self):
        self.fact = 0
        self.flops = 0.0
        self.solves = 0

    def __repr__(self):
        return f"Count(fact={self.fact}, solves={self.solves}, flops={self.flops:.3e})"


def to_band(A, b=None):
    """Lower-banded storage Ab[p, j] = A[j+p, j], p = 0..b."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if b is None:
        nz = np.nonzero(np.abs(A) > 0)
        b = int(np.max(np.abs(nz[0] - nz[1]))) if len(nz[0]) else 0
    Ab = np.zeros((b + 1, n))
    for p in range(b + 1):
        Ab[p, :n - p] = np.diagonal(A, -p)
    return Ab, b


def taylor_ldl_band(Ab, b, sigma, count=None, tiny=None):
    """One second-order-Taylor banded LDL^T pass of A - (sigma + eps) I.

    Returns (nu, s1, s2, ok) with nu the inertia count #{lambda < sigma}.
    `ok` is False if a pivot had to be perturbed (inertia then belongs to a
    matrix within `tiny` of A - sigma I, which is what LAPACK's Sturm codes
    also do).
    """
    n = Ab.shape[1]
    w = b + 1                                   # window edge
    if tiny is None:
        tiny = np.finfo(float).tiny ** 0.5

    # window W[c, p, q] = coefficient c of A[j+p, j+q] - (sigma+eps) delta
    W = np.zeros((3, w, w))
    for p in range(w):
        for q in range(p + 1):
            if p < n and q < n:
                W[0, p, q] = Ab[p - q, q]
                W[0, q, p] = W[0, p, q]
    for p in range(min(w, n)):
        W[0, p, p] -= sigma
        W[1, p, p] = -1.0

    d0 = np.empty(n)
    d1 = np.empty(n)
    d2 = np.empty(n)
    ok = True
    for j in range(n):
        a0 = W[0, 0, 0]
        if abs(a0) < tiny:
            a0 = tiny if a0 >= 0 else -tiny
            ok = False
        a1, a2 = W[1, 0, 0], W[2, 0, 0]
        d0[j], d1[j], d2[j] = a0, a1, a2

        m = min(b, n - 1 - j)
        if m > 0:
            w0 = W[0, 1:m + 1, 0]
            w1 = W[1, 1:m + 1, 0]
            w2 = W[2, 1:m + 1, 0]
            # reciprocal of the pivot as a Taylor series
            r0 = 1.0 / a0
            r1 = -a1 * r0 * r0
            r2 = (a1 * a1 - a0 * a2) * r0 * r0 * r0
            l0 = w0 * r0
            l1 = w0 * r1 + w1 * r0
            l2 = w0 * r2 + w1 * r1 + w2 * r0
            # t = d * l
            t0 = a0 * l0
            t1 = a0 * l1 + a1 * l0
            t2 = a0 * l2 + a1 * l1 + a2 * l0
            sl = slice(1, m + 1)
            W[0, sl, sl] -= np.outer(t0, l0)
            W[1, sl, sl] -= np.outer(t0, l1) + np.outer(t1, l0)
            W[2, sl, sl] -= np.outer(t0, l2) + np.outer(t1, l1) + np.outer(t2, l0)

        # slide the window one step
        W[:, :-1, :] = W[:, 1:, :]
        W[:, :, :-1] = W[:, :, 1:]
        W[:, -1, :] = 0.0
        W[:, :, -1] = 0.0
        jn = j + w                              # global index entering
        if jn < n:
            for q in range(w):
                gq = j + 1 + q
                if gq <= jn and gq < n:
                    p = jn - gq
                    v = Ab[p, gq] if p <= b else 0.0
                    W[0, w - 1, q] = v
                    W[0, q, w - 1] = v
            W[0, w - 1, w - 1] -= sigma
            W[1, w - 1, w - 1] = -1.0
            W[2, w - 1, w - 1] = 0.0

    nu = int(np.count_nonzero(d0 < 0))
    q1 = d1 / d0
    s1 = float(np.sum(q1))
    s2 = float(np.sum(q1 * q1 - 2.0 * d2 / d0))
    if count is not None:
        count.fact += 1
        # real banded LDL^T is n*b^2; Taylor-2 costs 6 mults per real mult
        count.flops += 6.0 * n * b * b
    return nu, s1, s2, ok


# --------------------------------------------------------------------------


def _laguerre_step(sigma, s1, s2, n, m, up):
    """Laguerre with parameter m.  m > 1 deliberately steps PAST m-1 roots,
    which is how the integer index enters the continuous map: from a point
    whose count is nu, aiming at index t means m = |t - nu| + 1."""
    if not (np.isfinite(s1) and np.isfinite(s2)):
        return None
    m = int(min(max(m, 1), n))
    disc = (n / m - 1.0) * (n * s2 - s1 * s1)
    if not np.isfinite(disc):
        return None
    rt = np.sqrt(max(disc, 0.0))
    best = None
    for sgn in (+1.0, -1.0):
        den = s1 + sgn * rt
        if den == 0.0:
            continue
        c = sigma - n / den
        if not np.isfinite(c):
            continue
        # Laguerre's safe branch is the LARGER |denominator|, i.e. the
        # SMALLEST step in the wanted direction.  Taking the other one
        # overshoots past many roots and destroys the method (measured).
        if up and c > sigma:
            best = c if best is None else min(best, c)
        if (not up) and c < sigma:
            best = c if best is None else max(best, c)
    return best


def laguerre_eigval(Ab, b, t, lo, hi, n, tol=1e-14, maxit=200, count=None,
                    scale=None, use_laguerre=True, verbose=False, certify=True):
    """Locate the t-th smallest eigenvalue (0-based) of the banded A.

    [lo, hi] must satisfy nu(lo) <= t < nu(hi).
    One factorization pass per iteration.  Returns (lam, mult, iters, width).
    """
    if scale is None:
        scale = max(abs(lo), abs(hi), 1.0)
    tiny = np.finfo(float).eps * scale
    nlo, *_ = taylor_ldl_band(Ab, b, lo, count, tiny=tiny)
    nhi, *_ = taylor_ldl_band(Ab, b, hi, count, tiny=tiny)
    assert nlo <= t < nhi, (nlo, t, nhi)

    sigma = 0.5 * (lo + hi)
    it = 0
    lam = sigma
    mu_est = 1
    side = 0
    flo = float(nlo - (t + 0.5))
    fhi = float(nhi - (t + 0.5))
    while it < maxit:
        it += 1
        nu, s1, s2, _ok = taylor_ldl_band(Ab, b, sigma, count, tiny=tiny)
        if nu <= t:
            lo, nlo = sigma, nu
            up = True                            # lambda_t lies above sigma
        else:
            hi, nhi = sigma, nu
            up = False
        lam = sigma
        if verbose:
            print(f"    it={it:3d} sig={sigma: .12f} nu={nu} "
                  f"w={hi-lo:.2e}")
        if not use_laguerre:                     # ablation: plain inertia bisection
            if hi - lo <= tol * scale:
                break
            sigma = 0.5 * (lo + hi)
            continue
        # PHASE 1 (index travel).  Laguerre-with-m advances the COUNT only a
        # few levels per pass (measured), so the moments are useless here.
        # What is not useless is that nu is monotone: false position on the
        # integer staircase, alternating with a bisection to keep the
        # halving guarantee.
        if nu not in (t, t + 1):
            if hi - lo <= tol * scale:
                break
            if up:                               # lo was replaced
                flo = nlo - (t + 0.5)
                fhi = fhi * 0.5 if side == +1 else float(nhi - (t + 0.5))
                side = +1
            else:
                fhi = nhi - (t + 0.5)
                flo = flo * 0.5 if side == -1 else float(nlo - (t + 0.5))
                side = -1
            den = fhi - flo
            sigma = ((lo * fhi - hi * flo) / den) if den != 0 else 0.5 * (lo + hi)
            if not (lo + 1e-6 * (hi - lo) < sigma < hi - 1e-6 * (hi - lo)):
                sigma = 0.5 * (lo + hi)
            continue
        # PHASE 2 (refinement).  The target is now the ADJACENT root.  The
        # moments also give its multiplicity for free: near a root of
        # multiplicity mu, s1 ~ mu/(sigma-lam) and s2 ~ mu/(sigma-lam)^2, so
        # mu ~ s1^2/s2.  Feeding that back as Laguerre's parameter restores
        # cubic convergence at a multiple root, where m=1 is only linear.
        mu = 1
        if np.isfinite(s1) and np.isfinite(s2) and s2 > 0:
            mu = int(min(max(round(s1 * s1 / s2), 1), max(1, n // 2)))
        mu_est = mu
        nxt = _laguerre_step(sigma, s1, s2, n, mu, up)
        if nxt is None:                          # step below one ulp
            lam = sigma
            break
        if abs(nxt - sigma) <= tol * scale:
            lam = nxt
            break
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)                # certified fallback
        sigma = nxt
    # certify the multiplicity: two extra passes bracketing the answer
    mult = mu_est
    if certify:
        d = max(8.0 * np.finfo(float).eps * scale, 1e-13 * scale)
        na, *_ = taylor_ldl_band(Ab, b, lam - d, count, tiny=tiny)
        nb_, *_ = taylor_ldl_band(Ab, b, lam + d, count, tiny=tiny)
        mult = max(1, nb_ - na)
    return lam, mult, it, hi - lo


def bisect_eigval(Ab, b, t, lo, hi, n, tol=1e-14, count=None, scale=None):
    return laguerre_eigval(Ab, b, t, lo, hi, n, tol=tol, maxit=400,
                           count=count, scale=scale, use_laguerre=False,
                           certify=True)


def eigvec_from_shift(A, lam, mult=1, iters=2, rng=None, count=None):
    """One (or two) inverse-iteration solves at the located shift.

    sigma is accurate to machine precision, so the solve is maximally
    ill-conditioned in exactly the right direction: one application already
    delivers the eigenvector.  Banded compiled solve.
    """
    import scipy.linalg as sla
    rng = np.random.default_rng(0) if rng is None else rng
    n = A.shape[0]
    Ab, b = to_band(A)
    ab = np.zeros((2 * b + 1, n))
    for p in range(b + 1):                       # full banded storage l=u=b
        ab[b - p, p:] = Ab[p, :n - p]            # superdiagonals
        ab[b + p, :n - p] = Ab[p, :n - p]        # subdiagonals
    ab[b, :] -= lam
    X = rng.standard_normal((n, mult))
    for _ in range(iters):
        X = sla.solve_banded((b, b), ab, X)
        X, _ = np.linalg.qr(X)
        if count is not None:
            count.solves += mult
    return X


def sweep_window(Ab, b, lo, hi, n, tol=1e-14, count=None, scale=None,
                 maxit=80, use_laguerre=True, deflate=True):
    """ALL eigenvalues in (lo, hi], in order, with a certified count.

    nu(lo) and nu(hi) certify k = nu(hi) - nu(lo) up front: no eigenvalue can
    be missed and none double-counted.  The eigenvalues are then taken in
    order, and the roots already found are removed from the MOMENTS rather
    than from the matrix -- Maehly deflation,

        s1 <- s1 - sum_j mu_j / (sigma - lam_j)
        s2 <- s2 - sum_j mu_j / (sigma - lam_j)^2,   n <- n - sum_j mu_j

    which is exact, costs O(k) scalars, and needs no orthogonalization and no
    matrix modification.  Without it the sweep is crippled: starting next to
    the root just found, Laguerre's own step is the distance to THAT root, so
    it climbs the gap by doubling -- 43 factorizations per eigenvalue,
    measured.  With it, ~5.
    """
    if scale is None:
        scale = max(abs(lo), abs(hi), 1.0)
    tiny = np.finfo(float).eps * scale
    tolabs = max(tol * scale, 2.0 * np.finfo(float).eps * scale)
    nlo, *_ = taylor_ldl_band(Ab, b, lo, count, tiny=tiny)
    nhi, *_ = taylor_ldl_band(Ab, b, hi, count, tiny=tiny)
    found, mults = [], []
    gap = 0.0
    t = nlo
    sigma = lo
    while t < nhi:
        it = 0
        lamb = None
        last_step = 0.0
        left, right = sigma, hi
        while it < maxit:
            it += 1
            nu, s1, s2, _ = taylor_ldl_band(Ab, b, sigma, count, tiny=tiny)
            if nu <= t:
                left = sigma
            else:
                right = sigma
                sigma = 0.5 * (left + right)
                continue
            ne = n
            if deflate:
                for lj, mj in zip(found, mults):
                    dl = sigma - lj
                    if dl != 0.0:
                        s1 -= mj / dl
                        s2 -= mj / (dl * dl)
                    ne -= mj
            ne = max(ne, 2)
            mu = 1
            if use_laguerre and np.isfinite(s1) and np.isfinite(s2) and s2 > 0:
                mu = int(min(max(round(s1 * s1 / s2), 1), max(1, ne // 2)))
            nxt = (_laguerre_step(sigma, s1, s2, ne, mu, True)
                   if use_laguerre else None)
            # A Laguerre step below one ulp comes back as None.  That is
            # CONVERGENCE, not failure -- reading it as failure and bisecting
            # the whole window instead costs 40+ wasted passes (measured).
            if use_laguerre and nxt is None:
                lamb = sigma
                break
            if nxt is not None and abs(nxt - sigma) <= tolabs:
                lamb = nxt
                last_step = abs(nxt - sigma)
                break
            if nxt is None or not (left < nxt < right):
                nxt = 0.5 * (left + right)
            last_step = abs(nxt - sigma)
            sigma = nxt
        if lamb is None:
            lamb = 0.5 * (left + right)
        # advance by the CERTIFIED count, never by an assumed multiplicity:
        # widen delta until nu(lam+delta) > t, so the count strictly grows and
        # a root can be neither re-found nor skipped.  Whatever ends up inside
        # (lam-delta, lam+delta) is a numerically indistinguishable cluster
        # and is reported as one eigenvalue of that multiplicity.
        d = max(8.0 * np.finfo(float).eps * scale, 4.0 * last_step, tolabs)
        grew = 0
        for _ in range(40):
            nb_, *_ = taylor_ldl_band(Ab, b, lamb + d, count, tiny=tiny)
            if nb_ > t:
                break
            d *= 2.0
            grew += 1
        # SAFETY NET.  If delta had to grow a long way, lamb was NOT actually
        # converged (Maehly deflation cancels catastrophically within a few
        # ulps of an already-deflated root, and the tiny bogus step it
        # produces reads as convergence).  The count stays consistent either
        # way, but the VALUE would be wrong by the width of the widened
        # bracket -- 1e-3 relative, measured.  Refine by bisection, which
        # cannot lie.
        if grew > 6:
            aa, bb2 = lamb, lamb + d
            while bb2 - aa > tolabs:
                mm = 0.5 * (aa + bb2)
                nm, *_ = taylor_ldl_band(Ab, b, mm, count, tiny=tiny)
                if nm <= t:
                    aa = mm
                else:
                    bb2 = mm
            lamb = 0.5 * (aa + bb2)
            d = max(8.0 * np.finfo(float).eps * scale, tolabs)
            for _ in range(40):
                nb_, *_ = taylor_ldl_band(Ab, b, lamb + d, count, tiny=tiny)
                if nb_ > t:
                    break
                d *= 2.0
        m = max(1, nb_ - t)
        if found:
            gap = lamb - found[-1]
        found.append(lamb)
        mults.append(m)
        t = max(t + m, nb_)
        # Stand off sqrt(eps)*||A|| from the root just deflated.  Closer than
        # that, the deflated pole mu/(sigma-lam) is known only to eps*||A||
        # in lam, so its subtraction leaves an error ~ eps*scale/(sigma-lam)^2
        # that swamps the surviving moment.
        # (Tried and discarded: warm-starting a fraction of the PREVIOUS gap
        # away.  It overshoots into the bisection fallback: 2.3x more passes.)
        sigma = min(lamb + max(d, np.sqrt(np.finfo(float).eps) * scale),
                    0.5 * (lamb + d + hi))
    return np.array(found), np.array(mults)


# ---------------------------------------------------------------------------
# VERDICT, re-measured independently of the agent that proposed this, run
# SEQUENTIALLY with nothing else executing. See GENERAL.md for the write-up.
#
#   As an EIGENPAIR solver it loses badly: 25-147x in flops and 317-1149x in
#   wall time against ARPACK shift-invert, because shift-invert amortizes ONE
#   factorization across all k eigenpairs while this refactorizes at every
#   shift (11.4 per eigenvalue, each ~6x a plain LDL^T for the Taylor-2 jet).
#
#   The ONE thing it wins is the certified WINDOW COUNT, below.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from ssj import window_count

    def lap2d(L):
        n = L * L
        A = np.zeros((n, n))
        for i in range(L):
            for j in range(L):
                k = i * L + j
                A[k, k] = 4.0
                if i + 1 < L:
                    A[k, k + L] = A[k + L, k] = -1.0
                if j + 1 < L:
                    A[k, k + 1] = A[k + 1, k] = -1.0
        return A

    def anderson2d(L, W, seed=0):
        r = np.random.default_rng(seed)
        n = L * L
        A = np.zeros((n, n))
        for i in range(L):
            for j in range(L):
                k = i * L + j
                A[k, k] = r.uniform(-W / 2, W / 2)
                if i + 1 < L:
                    A[k, k + L] = A[k + L, k] = -1.0
                if j + 1 < L:
                    A[k, k + 1] = A[k + 1, k] = -1.0
        return A

    def midgap(ev, frac):
        """A bound in the MIDDLE of a gap. Quantiles of an eigenvalue list are
        themselves eigenvalues, which makes 'how many in [lo, hi]' ambiguous to
        within rounding -- measured, that alone made BOTH methods disagree with
        the truth at 3 of 5 sizes. The question has to be posed well before it
        can be answered."""
        i = int(frac * len(ev))
        while i + 1 < len(ev) and ev[i + 1] - ev[i] < 1e-8:
            i += 1
        return 0.5 * (ev[i] + ev[i + 1])

    def t(f, r=3):
        f()
        best = float("inf")
        for _ in range(r):
            t0 = time.perf_counter()
            f()
            best = min(best, time.perf_counter() - t0)
        return best

    print("certified window count vs window_count (#20, purification)")
    for fam, mk in (("2D Laplacian", lap2d),
                    ("2D Anderson W=12", lambda L: anderson2d(L, 12, 0))):
        print(f"--- {fam}")
        print(f'{"N":>7}{"inertia":>10}{"window_count":>14}{"speedup":>9}'
              f'   counts (inertia/purify/true)')
        for L in (12, 16, 24, 32, 40):
            A = mk(L)
            ev = np.linalg.eigvalsh(A)
            lo, hi = midgap(ev, 0.25), midgap(ev, 0.75)
            Ab, b = to_band(A)

            def inert():
                a = taylor_ldl_band(Ab, b, lo)
                c = taylor_ldl_band(Ab, b, hi)
                return ((c[0] if isinstance(c, tuple) else c)
                        - (a[0] if isinstance(a, tuple) else a))

            ti, tw = t(inert), t(lambda: window_count(A, lo, hi))
            ci, cw = inert(), window_count(A, lo, hi)
            ct = int(np.sum((ev >= lo) & (ev <= hi)))
            print(f"{L * L:7}{ti * 1e3:9.1f}ms{tw * 1e3:13.1f}ms"
                  f"{tw / ti:8.1f}x   {ci}/{cw}/{ct}")

    print("\nwhere the certificate STOPS being certified: shifts near an "
          "eigenvalue")
    rng = np.random.default_rng(11)
    for fam, A in (("2D Laplacian L=24", lap2d(24)),
                   ("2D Anderson L=24", anderson2d(24, 12, 0))):
        ev = np.linalg.eigvalsh(A)
        n = len(ev)
        Ab, b = to_band(A)
        for label, gen in (
                ("generic shifts",
                 lambda: rng.uniform(ev[0] - 0.1, ev[-1] + 0.1)),
                ("within 1e-13..1e-9 of an eigenvalue",
                 lambda: ev[rng.integers(n)]
                 + rng.choice([-1, 1]) * 10 ** rng.uniform(-13, -9))):
            bad = worst = 0
            for _ in range(200):
                s = float(gen())
                out = taylor_ldl_band(Ab, b, s)
                nu = out[0] if isinstance(out, tuple) else out
                true = int(np.sum(ev < s))
                if nu != true:
                    bad += 1
                    worst = max(worst, abs(nu - true))
            print(f"  {fam:20} {label:36} wrong {bad:3}/200  max|dnu|={worst}")
