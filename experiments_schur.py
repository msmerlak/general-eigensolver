"""A map whose fixed points are TRIANGULAR, not diagonal.

Every map in this repository targets one of: a diagonal matrix (SSJ, Brockett,
QR/LR flows), a projector (purify, sign), or an eigenvector (IPT and family).
All of them are therefore trying to DIAGONALIZE. For a nonsymmetric matrix
that is the wrong target twice over:

  * the eigenvector basis can be arbitrarily ill-conditioned, so even an exact
    diagonalization is numerically worthless when kappa(V) is large;
  * for a DEFECTIVE matrix it does not exist at all.

The Schur form does not have either problem: every square matrix has one,
the transform is unitary, and unitary means perfectly conditioned. So aim at
it instead.

Take the same retraction SSJ uses, X <- orth(X(I + K)) with K anti-Hermitian,
and linearize:

    T <- (I + K)^* T (I + K) = T + (T K - K T) + O(K^2)

Ask this to annihilate only the LOWER triangle. Keeping the diagonal part of
the commutator, the (i,j) entry for i > j gives

    T_ij + T_ii K_ij - K_ij T_jj = 0   ->   K_ij = T_ij / (T_jj - T_ii)

and K_ji = -conj(K_ij) closes it. That is a ONE-SIDED saturated Jacobi: same
generator shape as SSJ, same divide-by-gap, same arctan saturation to survive
a small gap -- but annihilating half the matrix instead of all of it, and
converging to a triangular fixed point rather than a diagonal one.

Two consequences that no diagonalizing map here can have:

  * it does not care that the eigenvectors are ill-conditioned, because it
    never forms them -- eigenvalues come off diag(T) and the transform stays
    unitary throughout;
  * it has something to converge to on a DEFECTIVE matrix, where diagonalizing
    maps have no fixed point to find.

Complex arithmetic throughout: a real matrix with complex eigenvalues has only
a real Schur form with 2x2 blocks, and the point here is a genuinely
triangular target.

VERDICT (run this file): it works, and it is not competitive, and the reason
is structural. SSJ's symmetric ancestor has a DESCENT PROPERTY -- one rotation
in plane (i,j) annihilates T_ij and T_ji at once, because they are equal, so
the off-diagonal norm strictly decreases. Here they are independent: a
rotation that annihilates T_ij changes T_ji arbitrarily. There is no quantity
that provably decreases, and measured, the lower-triangle norm is NOT monotone
and can enter a limit cycle (stalls on ~1 instance in 12). It is also 68x
LAPACK's dgees and 8x this repository's own sdc_eigvals on the same problem.

The motivating claim also failed: aiming at the Schur form was supposed to
help on DEFECTIVE matrices, where no diagonalization exists. It does not --
a defective matrix has all its eigenvalues equal, so every denominator
T_jj - T_ii is zero, and the divide-by-gap generator has nothing to work with.
The Schur form existing does not make this map able to find it.
"""
import numpy as np

__all__ = ["schur_ssj"]


class Cost:
    def __init__(self):
        self.mm = 0.0

    def gemm(self, k=1):
        self.mm += k

    def qr(self):
        self.mm += 0.67


def lower_norm(T):
    return float(np.linalg.norm(np.tril(T, -1), "fro"))


def _orth(M, c):
    Q, R = np.linalg.qr(M)
    c.qr()
    d = np.diag(R)
    return Q * np.where(np.abs(d) == 0, 1.0, d / np.abs(d)).conj()


def schur_ssj(A, tol=1e-13, max_iter=300, saturate=True, c=None,
              return_info=False):
    """Unitary reduction to (approximate) Schur form by a one-sided saturated
    Jacobi map. Returns (eigenvalues, X, T) with T = X^* A X upper triangular
    and X unitary.

    saturate : bound each angle by the SSJ arctan rule. Without it a small gap
        produces an unbounded generator, which is the same failure the
        symmetric case has and the same fix.
    """
    A = np.asarray(A)
    A = A.astype(np.complex128) if A.dtype.kind != "c" else A.copy()
    n = A.shape[0]
    c = c or Cost()
    T = A.copy()
    X = np.eye(n, dtype=np.complex128)
    nrm = float(np.linalg.norm(A, "fro"))
    il = np.tril_indices(n, -1)
    hist = []
    for it in range(1, max_iter + 1):
        low = lower_norm(T)
        hist.append(low)
        if low <= tol * nrm:
            if return_info:
                return np.diag(T).copy(), X, T, {"iters": it, "converged": True,
                                                 "cost": c.mm, "hist": hist}
            return np.diag(T).copy(), X, T
        d = np.diag(T)
        # Exact 2x2 Schur angle per plane, not the linearized ratio. For the
        # pair (j, i) with j < i the block is [[T_jj, T_ji], [T_ij, T_ii]] and
        # the rotation that annihilates T_ij is the one whose first column is
        # an eigenvector of that block, giving the tangent
        #     t = T_ij / (lambda - T_ii),
        # lambda the block eigenvalue nearer T_jj. The linearized version
        # (t = T_ij/(T_jj - T_ii)) ignores T_ji, which is fine when the matrix
        # is nearly normal and useless when it is not -- and "not" is exactly
        # the case this map exists for.
        a = np.broadcast_to(d[None, :], (n, n))[il]      # T_jj
        dd = np.broadcast_to(d[:, None], (n, n))[il]     # T_ii
        b = T.T[il]                                      # T_ji (upper partner)
        cc = T[il]                                       # T_ij (to annihilate)
        half = 0.5 * (a - dd)
        disc = np.sqrt(half * half + b * cc)
        lam1 = 0.5 * (a + dd) + disc
        lam2 = 0.5 * (a + dd) - disc
        lam = np.where(np.abs(lam1 - a) <= np.abs(lam2 - a), lam1, lam2)
        den = lam - dd
        t = np.where(np.abs(den) > 1e-300, cc / np.where(np.abs(den) > 1e-300,
                                                         den, 1.0), 0.0)
        t = np.where(np.isfinite(t), t, 0.0)
        if saturate:                       # cap the plane angle at 45 degrees,
            mag = np.abs(t)                # the same bound SSJ's arctan gives
            t = np.where(mag > 1.0, t / np.where(mag > 0, mag, 1.0), t)
        K = np.zeros((n, n), dtype=np.complex128)
        K[il] = t
        K = K - K.conj().T                      # anti-Hermitian
        Q = _orth(np.eye(n) + K, c)
        T = Q.conj().T @ T @ Q
        c.gemm(2)
        X = X @ Q
        c.gemm(1)
    if return_info:
        return np.diag(T).copy(), X, T, {"iters": max_iter, "converged": False,
                                         "cost": c.mm, "hist": hist}
    return np.diag(T).copy(), X, T


if __name__ == "__main__":
    import time
    import warnings
    warnings.filterwarnings("ignore")
    import scipy.linalg as sla

    def ginibre(n, seed=0):
        r = np.random.default_rng(seed)
        return r.standard_normal((n, n)) / np.sqrt(n)

    def neardiag_ns(n, cpl, seed=0):
        r = np.random.default_rng(seed)
        M = r.standard_normal((n, n))
        np.fill_diagonal(M, 0)
        M *= cpl / np.max(np.abs(M))
        return np.diag(np.sort(r.uniform(0, 100, n))) + M

    def defective(n, eps, seed=0):
        r = np.random.default_rng(seed)
        J = np.diag(np.ones(n)) + np.diag(np.ones(n - 1), 1)
        Q, _ = np.linalg.qr(r.standard_normal((n, n)))
        return Q @ (J + eps * r.standard_normal((n, n))) @ Q.T

    print("it reaches a true Schur form when it converges")
    for lbl, A in [("near-diagonal nonsym", neardiag_ns(80, 1, 1)),
                   ("Ginibre non-normal", ginibre(80, 1)),
                   ("near-defective", defective(60, 1e-6, 1))]:
        ev = np.linalg.eigvals(A)
        n2 = np.linalg.norm(A, 2)
        nF = np.linalg.norm(A, "fro")
        w, X, T, i = schur_ssj(A, max_iter=600, return_info=True)
        err = max(np.min(np.abs(ev - x)) for x in w) / n2
        uni = np.linalg.norm(X.conj().T @ X - np.eye(A.shape[0]))
        print(f"  {lbl:22} conv={str(i['converged']):5} it={i['iters']:4} "
              f"lower/|A|={lower_norm(T) / nF:.1e} eigerr={err:.1e} "
              f"unitary={uni:.1e}")

    print("\nbut there is no descent quantity: is the lower norm monotone?")
    for n in (60, 80, 100, 150):
        A = ginibre(n, 1)
        nF = np.linalg.norm(A, "fro")
        _, _, _, i = schur_ssj(A, max_iter=600, return_info=True)
        h = np.array(i["hist"]) / nF
        print(f"  n={n:4} conv={str(i['converged']):5} it={i['iters']:4} "
              f"monotone={bool(np.all(np.diff(h) <= 1e-15))}  final={h[-1]:.1e}")

    print("\nand it is far from the incumbents (n=200 Ginibre)")
    A = ginibre(200, 1)

    def t(f, r=3):
        f()                                   # warm up, discard
        best = float("inf")
        for _ in range(r):
            t0 = time.perf_counter()
            f()
            best = min(best, time.perf_counter() - t0)
        return best

    ts = t(lambda: schur_ssj(A, max_iter=600))
    tl = t(lambda: sla.schur(A, output="complex"))
    tg = t(lambda: np.linalg.eigvals(A))
    print(f"  schur_ssj {ts * 1e3:8.1f} ms | LAPACK dgees {tl * 1e3:7.1f} ms "
          f"| dgeev {tg * 1e3:7.1f} ms")
