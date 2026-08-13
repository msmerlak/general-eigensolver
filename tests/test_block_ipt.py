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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
