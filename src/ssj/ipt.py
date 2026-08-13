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


def _ipt_iterate(W, d, V, max_iter, tol, norm_A, divergence_factor=1e3):
    """Run the IPT fixed point from V (diagonally normalized). Returns
    (V, Lambda, iters, converged, err) with err the relative residual proxy
    max|V_new - V_old| measured in the diagonal normalization.

    One gemm per iteration. Divergence is detected by the error growing past
    divergence_factor times its initial value, which is what the bounded basin
    looks like from inside the loop.
    """
    xp = _am(V)
    n = V.shape[0]
    idx = xp.arange(n)
    err0 = None
    err = np.inf
    for it in range(1, max_iter + 1):
        WV = W @ V                              # the single gemm
        Lam = d + xp.real(xp.diag(WV))          # Lambda_j = d_j + (WV)_jj
        gap = Lam[None, :] - d[:, None]         # gap_ij = Lambda_j - d_i
        # The diagonal of `gap` is (WV)_jj, not a level gap; the diagonal of V
        # is pinned to 1 anyway, so neutralize it rather than divide by it.
        gap = gap + xp.eye(n, dtype=gap.dtype)
        Vn = WV / gap
        Vn[idx, idx] = 1.0
        err = float(xp.max(xp.abs(Vn - V)))
        V = Vn
        if err0 is None:
            err0 = max(err, 1e-300)
        if err <= tol:
            return V, Lam, it, True, err
        if not np.isfinite(err) or err > divergence_factor * err0:
            return V, Lam, it, False, err
    return V, Lam, max_iter, False, err


def _finalize(A, V, Lam, xp):
    """Orthonormalize the diagonally-normalized V and return sorted (w, V).

    IPT's columns are exact eigenvectors up to scale at convergence, so they
    are already orthogonal in exact arithmetic; column normalization suffices
    and a QR pass repairs the roundoff-level defect. Eigenvalues are recomputed
    as Rayleigh quotients so they are consistent with the returned vectors.
    """
    V = V / xp.linalg.norm(V, axis=0, keepdims=True)
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
        W, d, V, max_iter, tol * norm_A, norm_A)
    w, V = _finalize(A, V, Lam, xp)
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
