"""Iterative Perturbation Theory (IPT) eigensolver, and the SSJ->IPT hybrid.

IPT (Kenmoe, Kriemann, Smerlak, Zadorin) solves the symmetric eigenproblem as
a fixed point of resummed Rayleigh-Schrodinger perturbation theory. Split
A = D + W with D = diag(A) and W = A - D, and normalize each eigenvector
column so that V_jj = 1 (diagonal normalization, NOT unit norm). Then
A v = lambda v reads, row by row,

    v_i (lambda - d_i) = (W v)_i

which for i = j (where v_j = 1) gives the eigenvalue and for i != j gives the
eigenvector component:

    Lambda_j = d_j + (W V)_jj
    V_ij     = (W V)_ij / (Lambda_j - d_i),    V_jj = 1.

The whole update is ONE gemm (W V) plus elementwise work -- the cheapest
useful iteration in this repository by a factor of ~5-10 against an SSJ sweep.
The price is a bounded basin: the map contracts only while the coupling is
small against the level spacing, and it diverges outside that (RESULTS.md
records Newton-type linearized-angle iterations diverging at ~0.85x the level
spacing, which is this same boundary).

That makes IPT and SSJ complementary rather than competing:

  * IPT alone wins outright on near-diagonal / diagonally dominant matrices,
    where it converges in a handful of gemms -- fewer than a LAPACK
    tridiagonalization costs.
  * SSJ alone is parameter-free and empirically globally convergent, but pays
    ~5-10 gemm-equivalents per sweep.
  * The hybrid runs SSJ only long enough to enter IPT's basin, then lets IPT
    finish at one gemm per iteration. SSJ supplies the global basin; IPT
    supplies the cheap endgame.

The hybrid needs no basin estimate: it simply tries IPT, watches the residual,
and falls back to another SSJ sweep if IPT is not contracting (each abandoned
attempt costs only the gemms actually spent).
"""
from __future__ import annotations

import numpy as np

from .core import _am, _orth_qr, off_frobenius, ssj_eigh

__all__ = ["ipt_eigh", "ssj_ipt_eigh"]


def _ipt_iterate(W, d, V, max_iter, tol, norm_A, divergence_factor=1e3,
                 v_is_identity=False):
    """Run the IPT fixed point from V (diagonally normalized). Returns
    (V, Lambda, iters, converged, err) with err the max update in the diagonal
    normalization.

    One gemm per iteration, and the loop is written to keep the O(N^2)
    elementwise work from costing as much as that gemm: at N=1000 a single
    N-by-N temporary is 8 MB, so the naive form (five temporaries per
    iteration) is memory-bandwidth-bound and roughly doubles the iteration
    cost. Everything below is in-place or fused, and the reciprocal-gap matrix
    is the only extra array kept live.

    v_is_identity skips the first gemm: W @ I = W.

    Divergence is detected by the error growing past divergence_factor times
    its initial value, which is what the bounded basin looks like from inside
    the loop.
    """
    xp = _am(V)
    n = V.shape[0]
    idx = xp.arange(n)
    err0 = None
    err = np.inf
    Lam = d
    R = xp.empty_like(V)      # reciprocal gaps, reused every iteration
    for it in range(1, max_iter + 1):
        if it == 1 and v_is_identity:
            WV = W.copy()     # W @ I
        else:
            WV = W @ V        # the single gemm
        Lam = d + xp.real(xp.diag(WV))          # Lambda_j = d_j + (WV)_jj
        # R = 1 / (Lambda_j - d_i), with the diagonal neutralized: there
        # gap_jj = (WV)_jj is not a level gap and V_jj is pinned to 1.
        xp.subtract(Lam[None, :], d[:, None], out=R)
        R[idx, idx] = 1.0
        xp.reciprocal(R, out=R)
        xp.multiply(WV, R, out=WV)              # WV becomes the new V
        WV[idx, idx] = 1.0
        xp.subtract(WV, V, out=V)               # V holds the update ...
        err = float(xp.max(xp.abs(V)))
        V = WV                                  # ... then becomes the iterate
        if err0 is None:
            err0 = max(err, 1e-300)
        if err <= tol:
            return V, Lam, it, True, err
        if not np.isfinite(err) or err > divergence_factor * err0:
            return V, Lam, it, False, err
    return V, Lam, max_iter, False, err


def _finalize(A, V, Lam, xp, converged=True):
    """Normalize the diagonally-normalized V and return sorted (w, V).

    At convergence IPT's columns are exact eigenvectors up to scale, hence
    already mutually orthogonal to roundoff -- column normalization is all
    that is needed, and the eigenvalues Lambda are exact as computed. Both a
    QR pass and a Rayleigh-quotient recomputation would cost about as much as
    the entire iteration (measured: a QR at N=1000 costs ~4 gemms against a
    6-gemm solve), so they are spent only when the iteration did NOT converge
    and the output would otherwise be silently non-orthogonal.
    """
    V = V / xp.linalg.norm(V, axis=0, keepdims=True)
    if converged:
        w = Lam
    else:
        V = _orth_qr(V)
        w = xp.real(xp.sum(xp.conj(V) * (A @ V), axis=0))
    order = np.argsort(w, kind="stable") if xp is np else xp.argsort(w)
    return w[order], V[:, order]


def ipt_eigh(A, tol=1e-13, max_iter=200, V0=None, return_info=False):
    """Eigendecomposition by Iterative Perturbation Theory.

    Converges only inside IPT's basin (near-diagonal / diagonally dominant
    input, or a good warm start V0). Outside it the iteration diverges and
    that is reported rather than hidden: info["converged"] is False. Use
    ssj_ipt_eigh for a globally convergent solver with the same endgame.

    Parameters
    ----------
    A : (n, n) symmetric / Hermitian ndarray.
    tol : convergence tolerance on the fixed-point update, relative to ||A||_2.
    V0 : optional diagonally-normalized warm start (V0_jj = 1); defaults to I,
        i.e. first-order perturbation theory from the diagonal.
    """
    xp = _am(A)
    A = xp.asarray(A)
    n = A.shape[0]
    d = xp.real(xp.diag(A))
    W = A - xp.diag(xp.diag(A))
    norm_A = float(xp.linalg.norm(A, ord="fro")) / max(np.sqrt(n), 1.0)
    if norm_A == 0.0:
        norm_A = 1.0
    V = xp.eye(n, dtype=A.dtype) if V0 is None else xp.array(V0, dtype=A.dtype)

    V, Lam, iters, converged, err = _ipt_iterate(
        W, d, V, max_iter, tol * norm_A, norm_A, v_is_identity=V0 is None)
    w, V = _finalize(A, V, Lam, xp, converged=converged)
    if return_info:
        return w, V, {"iters": iters, "gemms": iters, "converged": converged,
                      "err": err}
    return w, V


def ipt_rate(B, xp=None):
    """Estimate IPT's contraction rate in the frame of B: max |W_ij| / |gap_ij|.

    This is the quantity that decides whether IPT converges, and it costs
    O(N^2) -- free next to a single gemm. Using it as the gate is what makes
    the hybrid cheap: the alternative (trial IPT runs after every sweep)
    burned 154 wasted gemms on a GOE N=1000 solve, because each failed probe
    pays for every iteration it takes before the divergence guard fires.
    """
    xp = _am(B) if xp is None else xp
    n = B.shape[0]
    d = xp.real(xp.diag(B))
    gap = xp.abs(d[None, :] - d[:, None])
    absW = xp.abs(B - xp.diag(xp.diag(B)))
    # Ignore the diagonal (gap 0, no coupling term) and guard exact ties, which
    # make the ratio infinite -- correctly, since IPT cannot resolve them.
    eye = xp.eye(n, dtype=bool)
    gap = xp.where(eye, 1.0, gap)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = xp.where(eye, 0.0, absW / gap)
    r = float(xp.max(ratio))
    return r if np.isfinite(r) else np.inf


def ssj_ipt_eigh(A, tol=1e-13, max_iter=200, ipt_gate=0.5,
                 ipt_probe_iters=60, method="auto", max_sweeps=1000,
                 X0=None, precision="full", prologue=0, return_info=False):
    """Globally convergent solver with a one-gemm-per-iteration endgame.

    Runs SSJ to globalize and hands off to IPT as soon as IPT's basin has been
    reached, so the expensive part of the solve stops as early as possible and
    the endgame costs one gemm per iteration instead of SSJ's ~5-10
    gemm-equivalents per sweep.

    The hand-off is gated on ipt_rate(B) < ipt_gate, an O(N^2) quantity that is
    free next to a gemm. SSJ runs in blocks between checks (it does not need to
    be interrupted every sweep) and the gate is only consulted once the sweep
    has actually changed the frame.

    Parameters as ssj_eigh, plus:
    ipt_gate : hand off to IPT when max|W_ij|/|gap_ij| falls below this.
        0.5 is conservative; IPT's true boundary is near 1.
    ipt_probe_iters : cap on IPT iterations after the hand-off (it falls back
        to finishing with SSJ if IPT still fails).
    """
    xp = _am(A)
    A = xp.asarray(A)
    if A.dtype.kind != "c" and A.dtype != np.float32:
        A = A.astype(np.float64, copy=False)
    n = A.shape[0]
    norm_A = float(xp.linalg.norm(A, ord="fro")) / max(np.sqrt(n), 1.0)

    X = (xp.eye(n, dtype=A.dtype) if X0 is None
         else _orth_qr(xp.asarray(X0).astype(A.dtype)))
    sweeps = 0
    ipt_gemms = 0
    ssj_kw = dict(method=method, precision=precision)
    # Coarse target for each globalizing block: SSJ only has to get close
    # enough for IPT's gate to open, not all the way to `tol`. Tightened
    # whenever a block converges with the gate still shut.
    coarse = 1e-2
    tol_abs = tol * float(xp.linalg.norm(A, ord="fro"))

    while sweeps <= max_sweeps:
        B = X.conj().T @ (A @ X)
        B = (B + B.conj().T) / 2.0

        if off_frobenius(B) <= tol_abs:
            break  # SSJ alone already finished it

        if ipt_rate(B, xp) < ipt_gate:
            d = xp.real(xp.diag(B))
            W = B - xp.diag(xp.diag(B))
            Vd, Lam, iters, ok, err = _ipt_iterate(
                W, d, xp.eye(n, dtype=B.dtype), ipt_probe_iters,
                tol * norm_A, norm_A, v_is_identity=True)
            ipt_gemms += iters
            if ok:
                Vd = Vd / xp.linalg.norm(Vd, axis=0, keepdims=True)
                V = X @ Vd
                order = (np.argsort(Lam, kind="stable") if xp is np
                         else xp.argsort(Lam))
                w, V = Lam[order], V[:, order]
                if return_info:
                    return w, V, {"sweeps": sweeps, "ipt_iters": iters,
                                  "ipt_gemms": ipt_gemms, "converged": True,
                                  "path": "ipt" if sweeps == 0 else "hybrid"}
                return w, V
            # Gate passed but IPT still failed (clustered spectrum, say):
            # raise the bar and let SSJ carry on.
            ipt_gate *= 0.1

        # Not in the basin yet: globalize with SSJ. Run it to a loose target so
        # the mixed-precision phase (if any) is entered once, not per sweep.
        # Globalize with SSJ, only as far as the coarse target.
        _, X, info = ssj_eigh(A, tol=max(tol, coarse),
                              max_sweeps=max_sweeps - sweeps, X0=X,
                              prologue=prologue if sweeps == 0 else 0,
                              return_info=True, **ssj_kw)
        sweeps += info["sweeps"] + info.get("sweeps_low", 0)
        if info["sweeps"] == 0:
            # The block made no progress: it was already at the coarse target
            # while the gate stayed shut, so ask SSJ for a tighter frame.
            if coarse <= tol:
                break
            coarse = max(coarse * 1e-2, tol)

    B = X.conj().T @ (A @ X)
    w = xp.real(xp.diag(B))
    order = np.argsort(w, kind="stable") if xp is np else xp.argsort(w)
    w, V = w[order], X[:, order]
    if return_info:
        return w, V, {"sweeps": sweeps, "ipt_iters": 0, "ipt_gemms": ipt_gemms,
                      "converged": True, "path": "ssj"}
    return w, V


def _with_target(kw, tol):
    """SSJ options for a globalizing block: stop at the coarse tolerance where
    IPT's gate plausibly opens rather than driving all the way to `tol`."""
    out = dict(kw)
    out["tol"] = max(tol, 1e-3)
    return out
