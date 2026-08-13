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


def ssj_ipt_eigh(A, tol=1e-13, max_iter=200, ipt_probe_every=1,
                 ipt_probe_iters=12, method="auto", max_sweeps=1000,
                 X0=None, precision="full", prologue=0, return_info=False):
    """Globally convergent solver with a one-gemm-per-iteration endgame.

    Runs SSJ sweeps to globalize, probing after each sweep whether IPT's basin
    has been reached; once it has, IPT finishes the job at one gemm per
    iteration instead of SSJ's ~5-10 gemm-equivalents per sweep.

    The probe is the test: rather than estimating IPT's basin from coupling and
    gaps (fragile, and expensive to compute), the solver simply tries IPT for
    up to ipt_probe_iters iterations in the current Ritz frame and keeps the
    result if it converged. A failed probe costs only the gemms it actually
    used before the divergence guard fired.

    Parameters as ssj_eigh, plus:
    ipt_probe_every : probe after every k-th SSJ sweep (1 = every sweep).
    ipt_probe_iters : iterations allowed per probe.
    """
    xp = _am(A)
    A = xp.asarray(A)
    if A.dtype.kind != "c" and A.dtype != np.float32:
        A = A.astype(np.float64, copy=False)
    n = A.shape[0]

    # Cheap first attempt: if A is already near-diagonal, IPT alone solves it
    # and no SSJ sweep is ever run.
    X = xp.eye(n, dtype=A.dtype) if X0 is None else _orth_qr(xp.asarray(X0).astype(A.dtype))
    sweeps = 0
    ipt_gemms = 0

    while True:
        B = X.conj().T @ (A @ X)
        B = (B + B.conj().T) / 2.0
        norm_B = float(xp.linalg.norm(B, ord="fro")) / max(np.sqrt(n), 1.0)
        d = xp.real(xp.diag(B))
        W = B - xp.diag(xp.diag(B))

        if sweeps % ipt_probe_every == 0:
            Vd, Lam, iters, ok, err = _ipt_iterate(
                W, d, xp.eye(n, dtype=B.dtype), ipt_probe_iters,
                tol * norm_B, norm_B)
            ipt_gemms += iters
            if ok:
                # IPT converged in the current frame: compose and finish.
                Vd = Vd / xp.linalg.norm(Vd, axis=0, keepdims=True)
                V = _orth_qr(X @ Vd)
                w = xp.real(xp.sum(xp.conj(V) * (A @ V), axis=0))
                order = np.argsort(w, kind="stable") if xp is np else xp.argsort(w)
                w, V = w[order], V[:, order]
                if return_info:
                    return w, V, {"sweeps": sweeps, "ipt_iters": iters,
                                  "ipt_gemms": ipt_gemms, "converged": True,
                                  "path": "ipt" if sweeps == 0 else "hybrid"}
                return w, V

        if sweeps >= max_sweeps:
            break

        # Not in the basin yet: one more SSJ sweep to globalize.
        _, X, info = ssj_eigh(A, tol=tol, method=method, max_sweeps=1,
                              X0=X, precision=precision,
                              prologue=prologue if sweeps == 0 else 0,
                              return_info=True)
        sweeps += 1
        if info["converged"]:
            w = xp.real(xp.sum(xp.conj(X) * (A @ X), axis=0))
            order = np.argsort(w, kind="stable") if xp is np else xp.argsort(w)
            w, V = w[order], X[:, order]
            if return_info:
                return w, V, {"sweeps": sweeps, "ipt_iters": 0,
                              "ipt_gemms": ipt_gemms, "converged": True,
                              "path": "ssj"}
            return w, V

    w = xp.real(xp.sum(xp.conj(X) * (A @ X), axis=0))
    order = np.argsort(w, kind="stable") if xp is np else xp.argsort(w)
    w, V = w[order], X[:, order]
    if return_info:
        return w, V, {"sweeps": sweeps, "ipt_iters": 0, "ipt_gemms": ipt_gemms,
                      "converged": False, "path": "failed"}
    return w, V
