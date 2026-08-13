"""Block IPT: the fixed-point equation whose basin is a parameter."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import ipt_eig_partial  # noqa: E402
from ssj.block_ipt import block_ipt_eig, choose_block  # noqa: E402


def near_diagonal(n, coupling, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    W *= coupling / np.max(np.abs(W))
    return np.diag(np.arange(n, dtype=float)) + W


def test_block_extends_the_basin():
    """The headline: block IPT converges where plain IPT diverges."""
    n, tgt = 400, 200
    A = near_diagonal(n, 2.0)
    assert not ipt_eig_partial(A, [tgt], return_info=True)[2]["converged"]

    lam, v, info = block_ipt_eig(A, tgt, max_block=8, by="gap",
                                 return_info=True)
    assert info["converged"]
    exact = np.linalg.eigvals(A)
    scale = np.linalg.norm(A, 2)
    assert np.min(np.abs(exact - lam)) / scale < 1e-12
    assert np.linalg.norm(A @ v - lam * v) / scale < 1e-10


def test_bigger_block_extends_it_further():
    n, tgt = 400, 200
    A = near_diagonal(n, 8.0)
    small = block_ipt_eig(A, tgt, max_block=8, by="gap", return_info=True)[2]
    big = block_ipt_eig(A, tgt, max_block=32, by="gap", return_info=True)[2]
    assert not small["converged"] and big["converged"]


def test_agrees_with_plain_ipt_where_both_work():
    n, tgt = 300, 150
    A = near_diagonal(n, 0.2)
    w_plain, _ = ipt_eig_partial(A, [tgt])
    lam, _, info = block_ipt_eig(A, tgt, max_block=8, by="gap",
                                 return_info=True)
    assert info["converged"]
    assert abs(lam - w_plain[0]) < 1e-10 * np.linalg.norm(A, 2)


def test_block_always_contains_the_target():
    A = near_diagonal(100, 1.0)
    for by in ("gap", "ratio"):
        for mb in (1, 4, 16):
            B = choose_block(A, 42, max_block=mb, by=by)
            assert 42 in B and len(B) <= max(mb, 1)


def test_symmetric_case():
    rng = np.random.default_rng(2)
    n, tgt = 300, 150
    W = rng.standard_normal((n, n))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    A = np.diag(np.arange(n, dtype=float)) + 2.0 * W / np.max(np.abs(W))
    lam, v, info = block_ipt_eig(A, tgt, max_block=16, by="gap",
                                 return_info=True)
    assert info["converged"]
    exact = np.linalg.eigvalsh(A)
    assert np.min(np.abs(exact - lam.real)) / np.linalg.norm(A, 2) < 1e-11




def test_adaptive_stays_cheap_when_the_problem_is_easy():
    """The point of adapting: use b=1 when b=1 suffices, so easy problems cost
    what plain IPT costs instead of what a guessed block costs."""
    from ssj.block_ipt import adaptive_block_ipt_eig
    A = near_diagonal(300, 0.3)
    lam, v, info = adaptive_block_ipt_eig(A, 150, return_info=True)
    assert info["converged"] and info["block"] <= 2 and info["grew"] == 0
    exact = np.linalg.eigvals(A)
    assert np.min(np.abs(exact - lam)) / np.linalg.norm(A, 2) < 1e-12


def test_adaptive_grows_only_as_needed():
    from ssj.block_ipt import adaptive_block_ipt_eig
    easy = adaptive_block_ipt_eig(near_diagonal(300, 0.3), 150,
                                  return_info=True)[2]
    hard = adaptive_block_ipt_eig(near_diagonal(300, 2.0), 150,
                                  return_info=True)[2]
    assert hard["block"] > easy["block"]
    assert hard["converged"]


def test_adaptive_reaches_beyond_the_static_basin():
    """Adaptive converges where a static block of the same cap diverges,
    because it re-selects against the CURRENT lambda."""
    from ssj.block_ipt import adaptive_block_ipt_eig
    n, tgt = 300, 150
    A = near_diagonal(n, 8.0)
    static = block_ipt_eig(A, tgt, max_block=8, by="gap", return_info=True)[2]
    lam, v, ad = adaptive_block_ipt_eig(A, tgt, max_block=128,
                                        return_info=True)
    assert not static["converged"]
    assert ad["converged"]
    exact = np.linalg.eigvals(A)
    assert np.min(np.abs(exact - lam)) / np.linalg.norm(A, 2) < 1e-10


def test_adaptive_capped_block_fails_fast_rather_than_grinding():
    """With max_block capped for cost, a hopeless problem must give up
    cheaply so a caller can fall back, not spend the whole budget."""
    from ssj.block_ipt import adaptive_block_ipt_eig
    A = near_diagonal(300, 50.0)
    lam, v, info = adaptive_block_ipt_eig(A, 150, max_block=8, max_outer=40,
                                          return_info=True)
    assert not info["converged"] and info["block"] <= 8


def _sparse_at(coupling, n=2000, seed=5):
    import scipy.sparse as sp
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from bench_sparse import sparse_diagonally_dominant
    A0, d = sparse_diagonally_dominant(n, seed=seed)
    A = (sp.diags(d) + coupling * (A0 - sp.diags(d))).tocsr()
    return A, int(np.argmin(np.abs(d - np.median(d))))


def test_sparse_block_extends_the_sparse_basin():
    """The point of the sparse formulation: converge where plain sparse IPT
    diverges, verified against dense ground truth. Coupling 320 sits at
    rho ~ 0.38, well past the measured plain-IPT envelope."""
    from ssj.block_ipt import sparse_block_ipt_eig
    A, t = _sparse_at(320.0)
    _, _, plain = ipt_eig_partial(A, [t], return_info=True, hermitian=True,
                                  max_iter=400)
    assert not plain["converged"]          # plain IPT genuinely fails here

    lam, v, info = sparse_block_ipt_eig(A, t, return_info=True)
    assert info["converged"] and info["block"] > 1
    ev = np.linalg.eigvalsh(np.asarray(A.todense()))
    scale = np.linalg.norm(np.asarray(A.todense()), 2)
    assert np.min(np.abs(ev - np.real(lam))) / scale < 1e-11
    assert np.linalg.norm(A @ v - lam * v) / scale < 1e-9


def test_sparse_block_never_forms_a_submatrix():
    """The dense path builds A[C, C], which is 3.2 GB at n=20000. If this runs
    in reasonable memory the full-matvec formulation is real."""
    from ssj.block_ipt import sparse_block_ipt_eig
    A, t = _sparse_at(1.0, n=20000, seed=1)
    lam, v, info = sparse_block_ipt_eig(A, t, return_info=True)
    assert info["converged"]
    scale = float(np.max(np.abs(A.diagonal()))) + 1.0
    assert np.linalg.norm(A @ v - lam * v) / scale < 1e-10


def test_sparse_block_agrees_with_the_dense_implementation():
    """Same algorithm, different linear algebra: on a matrix small enough for
    both, they must find the same eigenvalue."""
    from ssj.block_ipt import adaptive_block_ipt_eig, sparse_block_ipt_eig
    A, t = _sparse_at(160.0, n=600, seed=3)
    Ad = np.asarray(A.todense())
    lam_s, _, info_s = sparse_block_ipt_eig(A, t, return_info=True)
    lam_d, _, info_d = adaptive_block_ipt_eig(Ad, t, return_info=True)
    assert info_s["converged"] and info_d["converged"]
    scale = np.linalg.norm(Ad, 2)
    assert abs(np.real(lam_s) - np.real(lam_d)) / scale < 1e-10


def test_hopeless_sparse_target_fails_cheaply():
    """Failure has to be cheap, or escalating before falling back is a net
    loss -- without the early give-up a hopeless target ran the whole outer
    budget at the largest block and turned a 1.1 s fallback into 4.9 s."""
    from ssj.block_ipt import sparse_block_ipt_eig
    A, t = _sparse_at(3000.0)
    _, _, info = sparse_block_ipt_eig(A, t, return_info=True, max_outer=100)
    assert not info["converged"]
    assert info["iters"] < 60              # gave up well inside the budget


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
