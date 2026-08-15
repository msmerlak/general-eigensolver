"""Accelerating the double-bracket flow.

Plain Brockett converges monotonically at ~0.9997 per step. A steady linear
rate is precisely what acceleration machinery is built for, so leaving it
there was premature. Four candidates, cheapest idea first:

  1. LINE SEARCH. The backtracking used before accepts any decrease and
     rescales tau by 1.6/0.4. The objective along a fixed direction is a
     smooth scalar function of tau, so fit it properly instead. Evaluating a
     trial is cheaper than it looks: off^2 = ||A||^2 - ||diag||^2 and ||A||_F
     is invariant on the orbit, so only diag(Q^T A Q) is needed, which is one
     gemm plus elementwise work rather than two gemms.

  2. MOMENTUM (heavy ball) in the Lie algebra, with transport. For a gradient
     method with rate 1 - kappa, momentum gives 1 - sqrt(kappa): at
     kappa = 3e-4 that is 0.9997 -> 0.983, about 40x fewer steps. The previous
     direction lives in the old basis, so it is transported, Omega_{k-1} ->
     Q^T Omega_{k-1} Q, before being reused.
     (BENCHMARKS.md records generator-space momentum FAILING for SSJ. That is
     not evidence here: SSJ is a saturated Newton map whose stabilizer
     momentum destroys, while this is a genuine gradient flow, where momentum
     is the textbook accelerator.)

  3. RIEMANNIAN CG, same transport, Polak-Ribiere.

  4. PRECONDITIONING N -- the interesting one. Near a diagonal fixed point the
     (i,j) mode decays like exp(-tau (n_j - n_i)(lambda_j - lambda_i) t), so
     tau is capped by the LARGEST such product while the slowest mode is set
     by the smallest: the rate is a ratio of gap-products, i.e. a condition
     number. Equalizing them means choosing the weight gaps INVERSELY
     proportional to the eigenvalue gaps,

         n_{i+1} - n_i  ~  1 / (lambda_{i+1} - lambda_i)

     and then Omega_ij = A_ij (n_j - n_i) ~ A_ij / (lambda_j - lambda_i),
     which is SSJ's generator. So this predicts SSJ IS the optimally
     preconditioned double-bracket flow, and a partial preconditioner should
     interpolate between the two. That is a testable claim, not a story.
"""
import numpy as np

__all__ = ["brockett_ls", "brockett_momentum", "brockett_cg",
           "brockett_precond"]


def off(A):
    return float(np.linalg.norm(A - np.diag(np.diag(A)), "fro"))


class Cost:
    def __init__(self):
        self.mm = 0.0

    def gemm(self, k=1):
        self.mm += k

    def qr(self):
        self.mm += 0.67


def _orth(M, c):
    Q, R = np.linalg.qr(M)
    c.qr()
    return Q * np.sign(np.where(np.diag(R) == 0, 1.0, np.diag(R)))


def _off_after(A, Omega, tau, c, nrm2):
    """off(Q^T A Q) for Q = orth(I + tau Omega), using only diag(Q^T A Q).

    off^2 = ||A||_F^2 - ||diag||^2 and ||A||_F is invariant on the orbit, so
    one gemm suffices per trial instead of the two a full conjugation needs.
    """
    n = A.shape[0]
    Q = _orth(np.eye(n) + tau * Omega, c)
    AQ = A @ Q
    c.gemm(1)
    dg = np.einsum("ij,ij->j", Q, AQ)
    return np.sqrt(max(nrm2 - float(dg @ dg), 0.0)), Q


def _apply(A, Q, c):
    out = Q.T @ A @ Q
    c.gemm(2)
    return out


def _search(A, Omega, tau, c, nrm2, o, trials=6):
    """Bracket-and-fit line search on tau. Returns (tau, Q, off)."""
    best = (None, None, o)
    t = tau
    seen = []
    for _ in range(trials):
        f, Q = _off_after(A, Omega, t, c, nrm2)
        seen.append((t, f, Q))
        if f < best[2]:
            best = (t, Q, f)
            t *= 2.0
        else:
            break
    if best[0] is None:                      # nothing worked: shrink
        t = tau
        for _ in range(trials):
            t *= 0.3
            f, Q = _off_after(A, Omega, t, c, nrm2)
            if f < o:
                return t, Q, f
        return None, None, o
    # one quadratic refinement around the best of the bracket
    seen.sort(key=lambda z: z[1])
    if len(seen) >= 3:
        (t1, f1, _), (t2, f2, _), (t3, f3, _) = seen[:3]
        den = (t1 - t2) * (f2 - f3) - (t2 - t3) * (f1 - f2)
        if abs(den) > 1e-300:
            tq = 0.5 * (((t1 ** 2 - t2 ** 2) * (f2 - f3)
                         - (t2 ** 2 - t3 ** 2) * (f1 - f2)) / den)
            if np.isfinite(tq) and tq > 0:
                f, Q = _off_after(A, Omega, tq, c, nrm2)
                if f < best[2]:
                    return tq, Q, f
    return best


def _weights(kind, A, n):
    if kind == "linear":
        return np.arange(n, dtype=float)
    d = np.sort(np.diag(A))                  # current eigenvalue estimates
    gaps = np.diff(d)
    gaps = np.where(np.abs(gaps) < 1e-300, 1e-300, gaps)
    N = np.concatenate([[0.0], np.cumsum(1.0 / gaps)])
    N = N / max(np.ptp(N), 1e-300) * n       # keep the scale comparable
    order = np.argsort(np.argsort(np.diag(A)))
    return N[order]


def _run(A, tol, max_iter, mode, beta=0.9, kind="linear", c=None):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    c = c or Cost()
    nrm = float(np.linalg.norm(A, "fro"))
    nrm2 = nrm ** 2
    tau = None
    prev_dir = None
    prev_g = None
    hist = []
    for it in range(1, max_iter + 1):
        o = off(A)
        hist.append(o)
        if o <= tol * nrm:
            return A, it, True, c, hist
        N = _weights(kind, A, n)
        G = A * (N[None, :] - N[:, None])            # the gradient direction
        if tau is None:
            tau = 1.0 / max(np.linalg.norm(G, "fro"), 1e-300)

        if mode == "ls" or prev_dir is None:
            D = G
        elif mode == "momentum":
            D = G + beta * prev_dir
        elif mode == "cg":
            num = float(np.sum(G * (G - prev_g)))    # Polak-Ribiere
            den = float(np.sum(prev_g * prev_g))
            gam = max(num / den, 0.0) if den > 0 else 0.0
            D = G + gam * prev_dir
        else:
            D = G

        tau_new, Q, f = _search(A, D, tau, c, nrm2, o)
        if Q is None:                                # direction unusable
            if mode in ("momentum", "cg") and prev_dir is not None:
                prev_dir = None                      # restart on the gradient
                continue
            return A, it, False, c, hist
        tau = tau_new
        A_new = _apply(A, Q, c)
        if mode in ("momentum", "cg"):
            prev_dir = Q.T @ D @ Q                   # transport to new basis
            c.gemm(2)
            prev_g = Q.T @ G @ Q
            c.gemm(2)
        A = A_new
    return A, max_iter, False, c, hist


def brockett_ls(A, tol=1e-12, max_iter=4000, c=None):
    return _run(A, tol, max_iter, "ls", c=c)


def brockett_momentum(A, tol=1e-12, max_iter=4000, beta=0.9, c=None):
    return _run(A, tol, max_iter, "momentum", beta=beta, c=c)


def brockett_cg(A, tol=1e-12, max_iter=4000, c=None):
    return _run(A, tol, max_iter, "cg", c=c)


def brockett_precond(A, tol=1e-12, max_iter=4000, mode="ls", c=None):
    return _run(A, tol, max_iter, mode, kind="precond", c=c)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from ssj import ssj_eigh

    def goe(n, seed=0):
        r = np.random.default_rng(seed)
        M = r.standard_normal((n, n))
        return (M + M.T) / np.sqrt(2 * n)

    def spread(n, seed=0):
        r = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(r.standard_normal((n, n)))
        return (Q * np.geomspace(1, 1000, n)) @ Q.T

    print("which accelerations help the double-bracket flow (n=50 GOE)")
    A = goe(50, 3)
    nrm = np.linalg.norm(A, "fro")
    for nm, fn in [("line search", brockett_ls),
                   ("momentum b=0.9", brockett_momentum),
                   ("CG (Polak-Ribiere)", brockett_cg),
                   ("preconditioned N", brockett_precond)]:
        B, it, ok, c, _ = fn(A.copy(), max_iter=4000)
        print(f"  {nm:22} conv={str(ok):5} steps={it:5} "
              f"matmul-eq={c.mm:8.0f} off/|A|={off(B) / nrm:.2e}")

    print("\nthe one that works, tuned, against SSJ")
    for lbl, M in [("GOE n=40 seed7", goe(40, 7)), ("GOE n=40 seed3", goe(40, 3)),
                   ("GOE n=60 seed1", goe(60, 1)),
                   ("well-separated n=40", spread(40, 3))]:
        B, it, ok, c, _ = brockett_momentum(M.copy(), beta=0.98, max_iter=8000)
        _, _, info = ssj_eigh(M, return_info=True)
        s = info["sweeps"]
        print(f"  {lbl:22} momentum: conv={str(ok):5} steps={it:5} "
              f"matmul-eq={c.mm:7.0f} | SSJ {s:3} sweeps ({s * 2.67:.0f}) "
              f"-> {c.mm / (s * 2.67):.0f}x")
