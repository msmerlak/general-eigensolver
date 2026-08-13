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

__all__ = ["ipt_eigh", "ipt_eig", "ipt_rate", "ssj_ipt_eigh", "refine_eig"]


def _ipt_iterate(W, d, V, max_iter, tol, norm_A, divergence_factor=1e3,
                 v_is_identity=False, hermitian=True):
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
        # Lambda_j = d_j + (WV)_jj. For a Hermitian problem the diagonal is
        # real by construction and taking the real part suppresses roundoff
        # drift; for a general matrix it is genuinely complex and must not be.
        diag_WV = xp.diag(WV)
        Lam = d + (xp.real(diag_WV) if hermitian else diag_WV)
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


def ipt_eig(A, tol=1e-13, max_iter=200, V0=None, return_info=False,
            sort=True):
    """Eigendecomposition of a GENERAL (nonsymmetric) square matrix by IPT.

    The IPT fixed point never used symmetry -- it needs only W @ V and
    diagonals -- so the iteration is unchanged from the symmetric case:

        Lambda_j = d_j + (W V)_jj,   V_ij = (W V)_ij / (Lambda_j - d_i)

    What changes is everything around it. Eigenvectors of a nonsymmetric
    matrix are not orthogonal, so the columns are only normalized, never
    reorthogonalized (doing so would be wrong, not merely wasteful), and no
    Rayleigh quotient is available: Lambda as computed IS the eigenvalue.

    The prize is larger here than in the symmetric case. LAPACK's dgeev
    (Hessenberg reduction plus QR iteration plus back-substitution) measured
    65 gemm-equivalents at N=1000 against dsyevd's 8.3, so a 4-6 gemm solve
    wins by a much wider margin -- when it converges.

    Convergence still requires being inside the basin, rho = max |W_ij| /
    |d_i - d_j| below ~1 (use ipt_rate). Two caveats specific to the general
    problem:

      * A real matrix can have complex eigenvalues. Real arithmetic cannot
        represent them, so pass A as complex when complex pairs are expected
        (the iteration itself is dtype-agnostic). In the near-diagonal regime
        this function targets -- well-separated real diagonal entries, small
        coupling -- the spectrum stays real by perturbation, which is exactly
        when IPT applies.
      * Eigenvalues are returned unsorted unless `sort`, which orders by real
        part then imaginary part; complex spectra have no canonical order.
    """
    xp = _am(A)
    A = xp.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if A.dtype.kind not in "cf":
        A = A.astype(np.float64)
    n = A.shape[0]
    d = xp.diag(A).copy()
    W = A - xp.diag(d)
    norm_A = float(xp.linalg.norm(A, ord="fro")) / max(np.sqrt(n), 1.0)
    if norm_A == 0.0:
        norm_A = 1.0
    V = xp.eye(n, dtype=A.dtype) if V0 is None else xp.array(V0, dtype=A.dtype)

    V, Lam, iters, converged, err = _ipt_iterate(
        W, d, V, max_iter, tol * norm_A, norm_A, v_is_identity=V0 is None,
        hermitian=False)

    V = V / xp.linalg.norm(V, axis=0, keepdims=True)
    w = Lam
    if sort:
        key = xp.real(w) if w.dtype.kind == "c" else w
        order = np.argsort(key, kind="stable") if xp is np else xp.argsort(key)
        w, V = w[order], V[:, order]
    if return_info:
        return w, V, {"iters": iters, "gemms": iters, "converged": converged,
                      "err": err}
    return w, V


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
    d = xp.diag(B)
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


def refine_eig(A, w0, V0, tol=1e-13, max_iter=50, return_info=False):
    """Refine an APPROXIMATE eigendecomposition to full precision with IPT.

    Given any approximate eigenpairs (w0, V0) of A -- from a float32 solve, a
    previous timestep, a reduced-order model, a perturbative estimate -- this
    changes basis into the approximate eigenframe, where A is near-diagonal by
    construction, and runs IPT there.

    Measured: a float32 LAPACK solve (eigenvalue error 6.4e-8) lands at
    rho = 7e-6, deep inside IPT's basin, and THREE iterations take it to
    7.3e-15 with residual 4.6e-14 -- full double precision.

    This is the general form of IPT's role: not only a solver for
    near-diagonal input, but a refinement engine for anything that produces an
    approximate eigenbasis. Note where it does NOT pay, because the arithmetic
    is unforgiving: refinement is cheap in ITERATIONS but the basis change
    costs an inverse and two gemms in COMPLEX arithmetic (~4x real), so it only
    wins when the presolve is genuinely much cheaper than a full solve. On this
    CPU stack it is not -- LAPACK's sgeev measured 0.905 s against dgeev's
    0.848 s, because dgeev is latency-bound in its sequential Hessenberg and QR
    sweeps rather than flop-bound, so halving precision buys nothing. The
    architecture pays where a cheap approximate solve genuinely exists:
    tensor-core hardware, tracking a slowly varying matrix, or any application
    that already has a nearby eigenbasis in hand.

    Returns (w, V) refined, or (w, V, info) with info["converged"],
    info["iters"] and info["rate"] (the measured IPT rate in the given frame,
    which is the honest diagnostic for whether the presolve was good enough).
    """
    xp = _am(A)
    A = xp.asarray(A)
    n = A.shape[0]
    cdtype = np.complex128 if A.dtype.itemsize > 4 else np.complex64
    Ac = A.astype(cdtype)
    V = xp.asarray(V0).astype(cdtype)

    B = xp.linalg.solve(V, Ac @ V)
    rate = ipt_rate(B, xp)
    w, Vd, info = ipt_eig(B, tol=tol, max_iter=max_iter, return_info=True,
                          sort=False)
    Vfull = V @ Vd
    Vfull = Vfull / xp.linalg.norm(Vfull, axis=0, keepdims=True)

    key = xp.real(w)
    order = np.argsort(key, kind="stable") if xp is np else xp.argsort(key)
    w, Vfull = w[order], Vfull[:, order]
    if return_info:
        return w, Vfull, {"converged": info["converged"], "iters": info["iters"],
                          "rate": rate}
    return w, Vfull
