"""The secular map: eigenvalues as roots of a scalar rational function.

Every map in this repository iterates on a vector, a subspace, a projector or
a matrix. This one iterates on a SCALAR, one per eigenvalue, and it is not a
fixed-point iteration at all -- it is root-finding on

    f(lambda) = 1 + sigma * sum_i z_i^2 / (d_i - lambda)

whose poles are exactly the diagonal entries of D and whose roots are exactly
the eigenvalues of A = D + sigma z z^T.

The connection to the rest of the repository is direct and is the reason this
is worth having. IPT expands (D + W)'s eigenvalues in a perturbation series
and needs rho < 1 for that series to converge. When W has rank one, the
secular equation is the EXACT RESUMMATION of that same series -- so it needs
no basin condition of any kind. It is correct at coupling 1e-6 and at coupling
1e+6 alike, which nothing else here can say.

Two structural facts make it cheap and safe:

  * Cauchy interlacing pins each root into its own open interval
    (d_i, d_{i+1}), so a bracketed solve cannot converge to the wrong root or
    miss one. There is no ordering ambiguity and no deflation guesswork.
  * f is monotone increasing on each such interval, so bisection is
    unconditionally convergent there.

Cost is O(n^2) for the whole eigendecomposition -- n roots, each found in
O(n) work per evaluation -- against O(n^3) for a dense symmetric solve. That
is a factor of n, not a constant.

The subtlety that decides whether this is usable is ORTHOGONALITY. Forming
eigenvectors directly as v_i ~ (D - lambda_i)^{-1} z loses orthogonality
badly when two roots are close, because the computed lambda differs from the
exact one by a rounding error that is large relative to the gap. The fix
(Gu-Eisenstat) is to discard the given z and recompute the one for which the
computed lambdas are EXACT roots, via Loewner's formula

    zhat_i^2 = prod_k (lambda_k - d_i) / prod_{k != i} (d_k - d_i)

and build the vectors from zhat. Both are implemented so the difference can
be measured rather than asserted.
"""
import numpy as np

__all__ = ["secular_eigh_rank1", "secular_eigh_lowrank"]


def _secular(lam, d, w, sigma):
    return 1.0 + sigma * np.sum(w / (d - lam))


def _roots(d, w, sigma, iters=60):
    """All n roots at once, by VECTORIZED bracketed bisection.

    Every root has its own bracket (d_i, d_{i+1}) by interlacing, so the n
    scalar solves are independent and can be advanced in lockstep: one
    n-by-n evaluation of f at all n midpoints per step. That replaces n
    Python-level root solves with `iters` numpy calls, which is where the
    O(n^2) flop count actually turns into O(n^2) time. 60 halvings exhaust
    double precision on any bracket.
    """
    n = len(d)
    span = sigma * float(np.sum(w))
    lo = d.copy()
    hi = np.empty(n)
    hi[:-1] = d[1:]
    hi[-1] = d[-1] + span
    a = np.nextafter(lo, hi)
    b = np.nextafter(hi, lo)

    def f(x):                       # f at each x_i, all i at once: O(n^2)
        return 1.0 + sigma * np.sum(w[:, None] / (d[:, None] - x[None, :]),
                                    axis=0)

    fa = f(a)
    # Plain bisection, and deliberately so: safeguarded Newton was tried here
    # and was WORSE on both counts -- accuracy fell from 1.6e-15 to 3.4e-13
    # (the quadratic step lands off the tightest bracket at the last digit)
    # and it was slower, because f and f' together need two n-by-n passes per
    # step against bisection's one, which more than cancels the fewer steps.
    for _ in range(iters):
        m = 0.5 * (a + b)
        stuck = (m <= a) | (m >= b)
        if np.all(stuck):
            break
        fm = f(m)
        same = (fm < 0) == (fa < 0)
        a = np.where(same & ~stuck, m, a)
        fa = np.where(same & ~stuck, fm, fa)
        b = np.where((~same) & ~stuck, m, b)
    return 0.5 * (a + b)


def _loewner(lam, d):
    """Recompute z so that the COMPUTED lam are its exact roots.

    Without this the eigenvectors lose orthogonality whenever two roots are
    close: v_i ~ (D - lam_i)^{-1} z is dominated by the entry whose pole is
    nearest, and a rounding-level error in lam_i is not small compared with
    (d_j - lam_i) there. With it, orthogonality is at machine precision by
    construction.

    Computed in LOG space: the products run over all n terms, so at n = 500
    with a large sigma the numerator overflows to inf and the ratio becomes
    nan. Loewner's theorem says the ratio is positive, so absolute values lose
    nothing and the logs are safe.
    """
    n = len(d)
    eps = 1e-323
    num = np.sum(np.log(np.abs(lam[None, :] - d[:, None]) + eps), axis=1)
    Dd = np.abs(d[None, :] - d[:, None])
    np.fill_diagonal(Dd, 1.0)                 # skip the k = i term
    den = np.sum(np.log(Dd + eps), axis=1)
    logz2 = num - den
    return np.exp(0.5 * np.clip(logz2, -700.0, 700.0))


def secular_eigh_rank1(d, z, sigma, fix=True, return_info=False):
    """Eigendecomposition of A = diag(d) + sigma * z z^T in O(n^2).

    fix : use the Loewner recomputation of z (Gu-Eisenstat). Turning it off
        shows what it is worth.

    Returns (w, V) with w ascending and V orthogonal, columns the eigenvectors.
    """
    d = np.asarray(d, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(d)
    flip = sigma < 0
    if flip:                                   # -A = (-D) + |sigma| z z^T
        d_use, s_use = -d[::-1], -sigma
        z_use = z[::-1]
    else:
        d_use, s_use, z_use = d, sigma, z
    order = np.argsort(d_use)
    d_s, z_s = d_use[order], z_use[order]

    lam = _roots(d_s, z_s ** 2, s_use)
    zz = _loewner(lam, d_s) * np.where(z_s < 0, -1.0, 1.0) if fix else z_s

    V = zz[:, None] / (d_s[:, None] - lam[None, :])
    V /= np.linalg.norm(V, axis=0, keepdims=True)

    Vfull = np.empty((n, n))
    Vfull[order, :] = V
    if flip:
        lam, Vfull = -lam[::-1], Vfull[::-1, ::-1]
    idx = np.argsort(lam)
    lam, Vfull = lam[idx], Vfull[:, idx]
    if return_info:
        return lam, Vfull, {"n": n}
    return lam, Vfull


def secular_eigh_lowrank(d, U, sig):
    """A = diag(d) + U diag(sig) U^T by SEQUENTIAL rank-one updates.

    Each update is exact, so the whole construction is exact -- there is still
    no basin condition. The cost, however, is O(n^3) per update because the
    accumulated eigenvector matrix has to be rotated, which is the honest
    reason this is a rank-one method and not a general low-rank one unless the
    vectors are kept in factored form.
    """
    d = np.asarray(d, dtype=float)
    U = np.asarray(U, dtype=float)
    n = len(d)
    lam = d.copy()
    V = np.eye(n)
    for k in range(U.shape[1]):
        zk = V.T @ U[:, k]
        idx = np.argsort(lam)
        lam_s, zk_s, Vs = lam[idx], zk[idx], V[:, idx]
        mu, Y = secular_eigh_rank1(lam_s, zk_s, float(sig[k]))
        lam, V = mu, Vs @ Y
    idx = np.argsort(lam)
    return lam[idx], V[:, idx]


if __name__ == "__main__":
    import time
    import warnings
    warnings.filterwarnings("ignore")

    def t(f, r=2):
        f()
        best = float("inf")
        for _ in range(r):
            t0 = time.perf_counter()
            f()
            best = min(best, time.perf_counter() - t0)
        return best

    print("exact at ANY coupling -- no basin, unlike every other map here")
    n = 500
    r = np.random.default_rng(0)
    d = np.sort(r.uniform(0, 10, n))
    z = r.standard_normal(n)
    for sigma in (1e-2, 1.0, 1e3, 1e8):
        A = np.diag(d) + sigma * np.outer(z, z)
        ev = np.linalg.eigvalsh(A)
        nrm = np.linalg.norm(A, 2)
        w, V = secular_eigh_rank1(d, z, sigma)
        print(f"  sigma={sigma:8g}  eig err={np.max(np.abs(w - ev)) / nrm:.1e}"
              f"  resid={np.max(np.linalg.norm(A @ V - V * w, axis=0)) / nrm:.1e}")

    print("\nwhat the Loewner recomputation buys (orthogonality)")
    for n in (200, 500):
        r = np.random.default_rng(0)
        d = np.sort(r.uniform(0, 10, n))
        z = r.standard_normal(n)
        _, V1 = secular_eigh_rank1(d, z, 1.0, fix=True)
        _, V0 = secular_eigh_rank1(d, z, 1.0, fix=False)
        print(f"  n={n:4}  with fix {np.linalg.norm(V1.T @ V1 - np.eye(n)):.1e}"
              f"   without {np.linalg.norm(V0.T @ V0 - np.eye(n)):.1e}")

    print("\nO(n^2) against dsyevd's O(n^3): the ratio improves with n")
    print(f'{"n":>6}{"secular":>11}{"dsyevd":>11}{"ratio":>8}{"eig err":>10}')
    for n in (500, 1000, 2000):
        r = np.random.default_rng(0)
        d = np.sort(r.uniform(0, 10, n))
        z = r.standard_normal(n)
        A = np.diag(d) + np.outer(z, z)
        ev = np.linalg.eigvalsh(A)
        ts = t(lambda: secular_eigh_rank1(d, z, 1.0))
        tl = t(lambda: np.linalg.eigh(A))
        w, _ = secular_eigh_rank1(d, z, 1.0)
        print(f"{n:6}{ts * 1e3:10.1f}ms{tl * 1e3:10.1f}ms{tl / ts:7.2f}x"
              f"{np.max(np.abs(w - ev)) / np.linalg.norm(A, 2):10.1e}")
