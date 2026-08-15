"""Path following: the eigenproblem as a continuation, not a fixed point.

Every map tried so far is a fixed-point iteration on a FIXED matrix, and they
all run into the same recorded wall (GENERAL.md):

    IPT's map is defined by A's own diagonal split, so its contraction rate
    rho(A) is a property of A, INDEPENDENT of the starting iterate. A warm
    start cannot rescue a divergent map -- the basis must actually change.

That closes off initialization as a remedy, but it also says exactly what
would work: change the MATRIX gradually and refresh the basis as you go.

Walk A(t) = D + t W from t = 0, where the eigenvectors are the identity and
the problem is free, to t = 1. At each step re-express in the current basis,

    B = V^T A(t) V

If V diagonalizes A(t - dt), then B is near-diagonal by CONSTRUCTION, so IPT
runs inside its basin whatever the coupling of the original A. The basin
condition stops being a property of the matrix and becomes a condition on the
step size dt -- and dt can be chosen by the very quantity that measures it,
rho(B), which costs O(n^2), free beside the gemms.

So this is a predictor-corrector path in matrix space whose fixed points exist
only at t = 1, controlled by the basin criterion itself. Structurally it is
not a fixed-point iteration at all, which is why it can go where they cannot.

The open question, and the reason this file measures rather than asserts, is
COST: each step costs 2 gemms for the rotation plus the corrector's gemms, so
the method wins only if the number of steps stays small. That is what the
experiment below settles.
"""
import numpy as np

__all__ = ["homotopy_eigh", "ipt_rate_of", "Cost"]


class Cost:
    """gemm-equivalents; QR charged 0.67 (4n^3/3 against a gemm's 2n^3)."""

    def __init__(self):
        self.mm = 0.0

    def gemm(self, k=1):
        self.mm += k

    def qr(self):
        self.mm += 0.67


def off(A):
    return float(np.linalg.norm(A - np.diag(np.diag(A)), "fro"))


def ipt_rate_of(B):
    """rho = max_{i != j} |B_ij| / |d_i - d_j|, the O(n^2) basin indicator."""
    d = np.diag(B)
    gap = np.abs(d[:, None] - d[None, :])
    np.fill_diagonal(gap, np.inf)
    W = np.abs(B - np.diag(d))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.max(np.where(gap > 0, W / gap, np.inf)))


def _ipt(B, c, tol=1e-13, max_iter=60):
    """Plain IPT on a near-diagonal symmetric B. Returns (w, U, converged).

    One gemm per iteration. U's columns are IPT's diagonally-normalized
    eigenvectors, normalized at the end; for a symmetric B at convergence they
    are orthogonal to roundoff, which is what keeps the accumulated V
    orthogonal without an explicit reorthogonalization every step.
    """
    n = B.shape[0]
    d = np.diag(B).copy()
    W = B - np.diag(d)
    scale = float(np.linalg.norm(B, "fro")) / max(np.sqrt(n), 1.0)
    V = np.eye(n)
    idx = np.arange(n)
    err0 = None
    for it in range(1, max_iter + 1):
        WV = W @ V
        c.gemm(1)
        lam = d + np.real(np.diag(WV))
        R = lam[None, :] - d[:, None]
        R[idx, idx] = 1.0
        np.reciprocal(R, out=R)
        Vn = WV * R
        Vn[idx, idx] = 1.0
        err = float(np.max(np.abs(Vn - V)))
        V = Vn
        if err0 is None:
            err0 = max(err, 1e-300)
        if err <= tol * scale:
            V = V / np.linalg.norm(V, axis=0, keepdims=True)
            return lam, V, True
        if not np.isfinite(err) or err > 1e3 * err0:
            break
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    return lam, V, False


def homotopy_eigh(A, gate=0.05, max_steps=2000, reorth_every=25, c=None,
                  loose=None, return_path=False):
    """Eigendecomposition by following A(t) = D + t W from t=0 to t=1.

    gate : the rate rho(B) allowed at each step. Smaller means more, easier
        steps; this is the whole step-size control and the only parameter.

    Returns (w, V) or (w, V, info) with info["steps"], info["cost"] in
    gemm-equivalents, and info["path"] the accepted t values.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    c = c or Cost()
    d0 = np.diag(A).copy()
    D = np.diag(d0)
    W = A - D
    V = np.eye(n)
    t = 0.0
    dt = 1.0
    path = [0.0]
    steps = 0
    w = d0
    while t < 1.0 - 1e-14 and steps < max_steps:
        t_try = min(t + dt, 1.0)
        At = D + t_try * W
        B = V.T @ (At @ V)
        c.gemm(2)
        rho = ipt_rate_of(B)
        if rho > gate and dt > 1e-12:
            dt *= 0.5                       # too far: the corrector would fail
            continue
        # intermediate steps only need a basis good enough to keep the next
        # rotation inside the basin; full accuracy is wasted until t = 1
        tol_k = 1e-13 if (loose is None or t_try >= 1.0 - 1e-14) else loose
        w, U, ok = _ipt(B, c, tol=tol_k)
        if not ok and dt > 1e-12:
            dt *= 0.5
            continue
        V = V @ U
        c.gemm(1)
        steps += 1
        if steps % reorth_every == 0:       # accumulated drift, cheap to fix
            V, _ = np.linalg.qr(V)
            c.qr()
        t = t_try
        path.append(t)
        dt *= 1.7                           # try to take a longer step next
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    if return_path:
        return w, V, {"steps": steps, "cost": c.mm, "path": path,
                      "reached": t}
    return w, V


if __name__ == "__main__":
    import os
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from ssj import ssj_eigh

    def neardiag(n, cpl, seed=0):
        r = np.random.default_rng(seed)
        M = r.standard_normal((n, n))
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        M *= cpl / np.max(np.abs(M))
        return np.diag(np.sort(r.uniform(0, 100, n))) + M

    def goe(n, seed=0):
        r = np.random.default_rng(seed)
        M = r.standard_normal((n, n))
        return (M + M.T) / np.sqrt(2 * n)

    def degenerate(n, seed=0):
        r = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(r.standard_normal((n, n)))
        lam = np.concatenate([np.ones(n // 2),
                              1 + 1e-9 * np.arange(n - n // 2)])
        return (Q * lam) @ Q.T

    print("It globalizes IPT -- and costs far more than SSJ, which already does")
    print(f'{"case":18}{"rho(A)":>10}{"steps":>7}{"gemm-eq":>9}{"err":>10}'
          f'{"  | SSJ":>8}{"ratio":>8}')
    cases = [("near-diag 0.5", neardiag(200, 0.5, 1)),
             ("near-diag 5", neardiag(200, 5, 1)),
             ("near-diag 50", neardiag(200, 50, 1)),
             ("GOE", goe(200, 1)),
             ("degenerate", degenerate(150, 2))]
    for lbl, A in cases:
        ev = np.sort(np.linalg.eigvalsh(A))
        nrm = np.linalg.norm(A, 2)
        w, V, i = homotopy_eigh(A, gate=1.0, max_steps=4000, return_path=True)
        err = np.max(np.abs(np.sort(w) - ev)) / nrm
        _, _, si = ssj_eigh(A, return_info=True)
        sc = si["sweeps"] * 2.67
        print(f'{lbl:18}{ipt_rate_of(A):10.0f}{i["steps"]:7}{i["cost"]:9.0f}'
              f'{err:10.1e}{sc:8.0f}{i["cost"] / sc:7.0f}x')
    print("\nthe last row is the point: exact degeneracy gives a ZERO gap, and")
    print("no step size makes a zero denominator finite. SSJ's arctan does.")
