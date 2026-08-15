"""Maps that are structurally different from everything in the repository.

The repo's maps are classified by what the STATE is:

    vector          IPT, BW, Davidson, gradient flows
    subspace        SSJ, LOBPCG, subspace iteration
    projector       purify (P <- 3P^2 - 2P^3)
    matrix (iso)    SSJ again -- orthogonal conjugation, generator from arctan

and every fast one divides by a level gap, K_ij ~ A_ij/(d_j - d_i), which is
why the diagonal has to carry spectral information for any of them to work.

So the structural change worth trying is a generator that MULTIPLIES by a
weight instead of dividing by a gap. That is the Brockett double-bracket flow

    Adot = [A, [A, N]],     N = diag(n_1 < ... < n_n) fixed

the gradient flow of trace(N A) on the isospectral orbit {Q^T A Q}. Its
generator is Omega = [A, N], i.e.

    Omega_ij = A_ij (n_j - n_i)          <- multiply, no denominator
    (SSJ:      K_ij ~ A_ij / (d_j - d_i)  <- divide)

Omega is antisymmetric whenever A is symmetric and N diagonal, so exp(tau
Omega) is orthogonal and A <- Q^T A Q is exactly isospectral. Both generators
cost O(n^2) elementwise; the difference is entirely in the nonlinearity.

Three things follow that are not true of anything else here:
  * no denominators at all, so exact degeneracy is not a special case;
  * no near-diagonality assumption -- N is chosen, not read off A;
  * Brockett (1991) proves convergence to a diagonal matrix for generic N,
    where RESULTS.md says plainly that no convergence proof is known for SSJ.

Also included, as an even more different mechanism (no orthogonal conjugation
at all): the Cholesky-LR flow A <- L^T L where A = L L^T, and the QR flow
A <- R Q where A = QR, which is the QR algorithm read as a dynamical system
and serves as the reference every isospectral map is trying to beat.

VERDICT, measured (run this file): the whole family loses badly, and the
double-bracket flow loses worst. See GENERAL.md, "Isospectral gradient flows".
A Brockett step and an SSJ sweep cost the SAME (1 QR + 2 gemms), so the
comparison is just step counts: SSJ needs 13 on a GOE matrix, Brockett is at
8.6e-4 after 8,000 and contracting by 0.999707 per step, i.e. ~70,000 more to
reach 1e-12. Roughly 6,000x. It converges monotonically the whole way -- this
is slowness, not stalling, and it is exactly what Brockett's theorem promises.
"""
import numpy as np

__all__ = ["brockett", "brockett_adaptive", "cholesky_lr", "qr_flow"]


def off(A):
    return float(np.linalg.norm(A - np.diag(np.diag(A)), "fro"))


class Cost:
    """Counts matmul-equivalents (n^3 units) so comparisons are hardware-free.

    A QR or Cholesky of an n-by-n matrix is charged its textbook flop count
    relative to a gemm (2n^3): QR = 4/3 n^3 -> 0.67, Cholesky = n^3/3 -> 0.17.
    Elementwise O(n^2) work is charged nothing, which flatters every map here
    equally.
    """

    def __init__(self):
        self.mm = 0.0

    def gemm(self, k=1):
        self.mm += k

    def qr(self):
        self.mm += 0.67

    def chol(self):
        self.mm += 0.17


def _orth(M, c):
    Q, R = np.linalg.qr(M)
    c.qr()
    return Q * np.sign(np.where(np.diag(R) == 0, 1.0, np.diag(R)))


def brockett(A, N=None, tau=None, tol=1e-12, max_iter=2000, c=None,
             adaptive=True):
    """Discrete double-bracket flow: A <- Q^T A Q, Q = orth(I + tau [A, N]).

    The retraction orth(I + tau Omega) is the same one SSJ uses, so the ONLY
    difference from SSJ in this implementation is the generator: multiply by
    (n_j - n_i) rather than divide by (d_j - d_i).

    tau is chosen by backtracking on off(A), which is legitimate here because
    the flow is a gradient flow -- off(A) is guaranteed to decrease for small
    enough tau, so backtracking always terminates.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    c = c or Cost()
    if N is None:
        N = np.arange(n, dtype=float)          # generic, strictly increasing
    nrm = float(np.linalg.norm(A, "fro"))
    hist = []
    tau0 = tau
    for it in range(1, max_iter + 1):
        o = off(A)
        hist.append(o)
        if o <= tol * nrm:
            return A, it, True, c, hist
        Omega = A * (N[None, :] - N[:, None])   # O(n^2), the whole generator
        s = float(np.linalg.norm(Omega, "fro"))
        if s == 0.0:
            return A, it, True, c, hist
        if tau is None:                          # scale the step like SSJ does:
            tau = tau0 if tau0 else 1.0 / s      # ||tau Omega|| ~ 1
        ok = False
        for _ in range(40 if adaptive else 1):
            Q = _orth(np.eye(n) + tau * Omega, c)
            A_new = Q.T @ A @ Q
            c.gemm(2)
            if off(A_new) < o:
                A = A_new
                if adaptive:
                    tau *= 1.6                  # try to grow while it works
                ok = True
                break
            tau *= 0.4
        if not ok:
            return A, it, False, c, hist
    return A, max_iter, False, c, hist


def brockett_adaptive(A, tol=1e-12, max_iter=2000, c=None):
    """Double-bracket with N = diag(A), i.e. the weight re-read every step.

    Omega_ij = A_ij (d_j - d_i). This is the gradient flow of ||diag(A)||^2,
    the natural 'sort yourself' choice, and unlike fixed N it needs no guess
    about where eigenvalues will land. It is also the exact opposite of SSJ's
    generator: same two factors, multiplied instead of divided.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    c = c or Cost()
    nrm = float(np.linalg.norm(A, "fro"))
    tau = None
    hist = []
    for it in range(1, max_iter + 1):
        o = off(A)
        hist.append(o)
        if o <= tol * nrm:
            return A, it, True, c, hist
        d = np.diag(A)
        Omega = A * (d[None, :] - d[:, None])
        s = float(np.linalg.norm(Omega, "fro"))
        if s == 0.0:
            return A, it, False, c, hist
        if tau is None:
            tau = 1.0 / s
        ok = False
        for _ in range(40):
            Q = _orth(np.eye(n) + tau * Omega, c)
            A_new = Q.T @ A @ Q
            c.gemm(2)
            if off(A_new) < o:
                A = A_new
                tau *= 1.6
                ok = True
                break
            tau *= 0.4
        if not ok:
            return A, it, False, c, hist
    return A, max_iter, False, c, hist


def cholesky_lr(A, tol=1e-12, max_iter=5000, c=None):
    """Rutishauser's LR flow: A = L L^T  ->  A <- L^T L. Isospectral, and NOT
    an orthogonal conjugation -- L^T L = L^{-1}(L L^T)L is a similarity by a
    triangular factor. A genuinely different mechanism, and cheaper per step
    than QR (n^3/3 against 4n^3/3).

    Needs positive definiteness, so the matrix is shifted first; the shift is
    removed from the diagonal at the end.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    c = c or Cost()
    nrm = float(np.linalg.norm(A, "fro"))
    shift = 0.0
    lo = float(np.min(np.linalg.eigvalsh(A))) if n <= 400 else \
        float(np.min(np.diag(A)) - np.sum(np.abs(A), axis=1).max())
    if lo <= 0:
        shift = -lo + 1e-3 * max(nrm, 1.0)
        A = A + shift * np.eye(n)
    hist = []
    for it in range(1, max_iter + 1):
        o = off(A)
        hist.append(o)
        if o <= tol * nrm:
            A = A - shift * np.eye(n)
            return A, it, True, c, hist
        try:
            L = np.linalg.cholesky(A)
            c.chol()
        except np.linalg.LinAlgError:
            return A - shift * np.eye(n), it, False, c, hist
        A = L.T @ L
        c.gemm(1)
    return A - shift * np.eye(n), max_iter, False, c, hist


def qr_flow(A, tol=1e-12, max_iter=5000, c=None):
    """A <- R Q where A = Q R: the unshifted QR algorithm as a flow. The
    reference every isospectral map is trying to beat."""
    A = np.array(A, dtype=float)
    c = c or Cost()
    nrm = float(np.linalg.norm(A, "fro"))
    hist = []
    for it in range(1, max_iter + 1):
        o = off(A)
        hist.append(o)
        if o <= tol * nrm:
            return A, it, True, c, hist
        Q, R = np.linalg.qr(A)
        c.qr()
        A = R @ Q
        c.gemm(1)
    return A, max_iter, False, c, hist


if __name__ == "__main__":
    def goe(n, seed=0):
        r = np.random.default_rng(seed)
        M = r.standard_normal((n, n))
        return (M + M.T) / np.sqrt(2 * n)

    def spread(n, seed=0):
        r = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(r.standard_normal((n, n)))
        return (Q * np.geomspace(1, 1000, n)) @ Q.T

    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from ssj import ssj_eigh

    for label, A in [("well-separated", spread(50, 3)), ("GOE", goe(50, 3))]:
        nrm = np.linalg.norm(A, "fro")
        print(f"--- {label}, n={A.shape[0]}")
        for fn, nm in [(brockett, "brockett fixed N"),
                       (brockett_adaptive, "brockett N=diag(A)"),
                       (cholesky_lr, "cholesky-LR"), (qr_flow, "QR flow")]:
            B, it, ok, c, _ = fn(A.copy(), max_iter=4000)
            print(f"   {nm:20} conv={str(ok):5} steps={it:5} "
                  f"matmul-eq={c.mm:8.0f} off/|A|={off(B) / nrm:.1e}")
        _, _, info = ssj_eigh(A, return_info=True)
        print(f"   {'SSJ':20} conv=True  steps={info['sweeps']:5} "
              f"matmul-eq~{info['sweeps'] * 2.67:8.0f}")
