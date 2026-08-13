"""Large sparse interior eigenpairs: where IPT's advantage is structural.

The repository's earlier partial-solver wins were on DENSE matrices, where
ARPACK's shift-invert factorization is O(N^3/3) but affordable, so the margin
came from IPT's low iteration count and topped out around 4-123x.

On LARGE SPARSE matrices the asymmetry becomes structural rather than
constant-factor. Targeting interior eigenvalues with a Krylov method requires
shift-invert, i.e. factorizing (A - sigma I) -- and for sparsity with no good
elimination ordering (a random graph, as opposed to a lattice or a banded
matrix) the LU fill-in explodes. Measured on this family, fill-in as a
multiple of the original nnz:

    N=2000   88.7x        N=10000  430.7x
    N=5000  222.1x        N=20000  factorization no longer affordable

IPT needs no factorization at all: 3-5 iterations, each one sparse matvec,
so its cost is O(nnz) essentially independent of N.

Run: python3 bench_sparse.py
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from scipy.sparse.linalg import (LinearOperator, eigs, eigsh,  # noqa: E402
                                  lobpcg, splu)

from ssj import ipt_eig_partial  # noqa: E402


def sparse_diagonally_dominant(n, nnz_row=8, coupling=0.05, seed=0):
    """Random sparse graph, wide diagonal spread, weak off-diagonal coupling.

    Models a configuration-interaction / random-network Hamiltonian: diagonal
    energies spread over a wide range, sparse weak coupling between them. The
    random (non-geometric) sparsity is the point -- it has no good elimination
    ordering, so it is the honest hard case for shift-invert.
    """
    rng = np.random.default_rng(seed)
    d = rng.uniform(0, float(n), n)
    m = nnz_row * n
    rows = rng.integers(0, n, m)
    cols = rng.integers(0, n, m)
    vals = rng.standard_normal(m) * coupling
    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W = (W + W.T) * 0.5
    W = W.tolil()
    W.setdiag(0)
    return (W.tocsr() + sp.diags(d)).tocsr(), d


def sparse_nonsymmetric(n, nnz_row=8, coupling=0.05, seed=0):
    """The same family without symmetrization. Harder in practice: for
    nonsymmetric sparse interior targets there is no LOBPCG equivalent in
    scipy, so shift-invert (scipy.sparse.linalg.eigs with sigma) is
    essentially the only alternative -- and its fill-in is WORSE here than in
    the symmetric case (measured 116.6x nnz at N=2000 rising to 568.3x at
    N=10000)."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0, float(n), n)
    m = nnz_row * n
    rows = rng.integers(0, n, m)
    cols = rng.integers(0, n, m)
    vals = rng.standard_normal(m) * coupling
    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W = W.tolil()
    W.setdiag(0)
    return (W.tocsr() + sp.diags(d)).tocsr(), d


def _lobpcg_interior(A, sigma, cols, n, tol=1e-10):
    """The fair matvec-only competitor: LOBPCG on (A - sigma I)^2, which
    targets the interior without any factorization -- at the cost of squaring
    the conditioning, which is what makes it need many more iterations."""
    Ash = (A - sigma * sp.eye(n)).tocsr()
    Op = LinearOperator((n, n), matvec=lambda X: Ash @ (Ash @ X), dtype=float)
    dd = np.asarray(A.diagonal()).ravel()
    scale = np.maximum((dd - sigma) ** 2, 1e-12)
    Minv = LinearOperator(
        (n, n),
        matvec=lambda X: X / (scale[:, None] if X.ndim > 1 else scale),
        dtype=float,
    )
    X0 = np.zeros((n, len(cols)))
    X0[cols, np.arange(len(cols))] = 1.0
    w, V = lobpcg(Op, X0, M=Minv, largest=False, tol=tol, maxiter=500)
    wA = np.array([float(V[:, i] @ (A @ V[:, i])) / float(V[:, i] @ V[:, i])
                   for i in range(V.shape[1])])
    return wA, V


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [2000, 5000, 10000, 20000]
    print(f'{"N":>7} {"nnz":>8} {"IPT":>10} {"ARPACK s-i":>12} {"LU fill":>9} '
          f'{"LOBPCG":>10} {"IPT vs best":>12}')
    for n in sizes:
        A, d = sparse_diagonally_dominant(n)
        order = np.argsort(np.abs(d - np.median(d)))
        cols = list(order[:4])
        sigma = float(np.mean(d[cols]))
        nrm = float(np.max(np.abs(d))) + 1.0

        t0 = time.perf_counter()
        w, V, info = ipt_eig_partial(A, cols, return_info=True, hermitian=True)
        ti = time.perf_counter() - t0
        resid = max(np.linalg.norm(A @ V[:, j] - w[j] * V[:, j])
                    for j in range(len(cols))) / nrm

        ta, fill = float("nan"), float("nan")
        if n <= 10000:
            try:
                lu = splu((A - sigma * sp.eye(n)).tocsc())
                fill = (lu.L.nnz + lu.U.nnz) / A.nnz
                del lu
                t0 = time.perf_counter()
                eigsh(A, k=4, sigma=sigma)
                ta = time.perf_counter() - t0
            except Exception:
                ta = float("inf")

        try:
            t0 = time.perf_counter()
            _lobpcg_interior(A, sigma, cols, n)
            tl = time.perf_counter() - t0
        except Exception:
            tl = float("inf")

        best = min(x for x in (ta, tl) if not np.isnan(x))
        print(f'{n:>7} {A.nnz:>8} {ti:>9.4f}s {ta:>11.3f}s {fill:>8.1f}x '
              f'{tl:>9.3f}s {best/ti:>11.0f}x')
        print(f'        IPT {info["iters"]} its, relative residual {resid:.1e}',
              flush=True)

    print()
    print("NONSYMMETRIC (no LOBPCG equivalent exists; shift-invert is the "
          "only alternative)")
    print(f'{"N":>7} {"IPT":>10} {"its":>4} {"ARPACK eigs s-i":>16} '
          f'{"LU fill":>9} {"speedup":>9}')
    for n in [s_ for s_ in sizes if s_ <= 10000]:
        A, d = sparse_nonsymmetric(n)
        cols = list(np.argsort(np.abs(d - np.median(d)))[:4])
        sigma = float(np.mean(d[cols]))
        nrm = float(np.max(np.abs(d))) + 1.0
        t0 = time.perf_counter()
        w, V, info = ipt_eig_partial(A, cols, return_info=True, hermitian=False)
        ti = time.perf_counter() - t0
        ta, fill = float("nan"), float("nan")
        try:
            lu = splu((A - sigma * sp.eye(n)).tocsc())
            fill = (lu.L.nnz + lu.U.nnz) / A.nnz
            del lu
            t0 = time.perf_counter()
            eigs(A, k=4, sigma=sigma)
            ta = time.perf_counter() - t0
        except Exception:
            ta = float("inf")
        print(f'{n:>7} {ti:>9.4f}s {info["iters"]:>4} {ta:>15.3f}s '
              f'{fill:>8.1f}x {ta/ti:>8.0f}x', flush=True)
