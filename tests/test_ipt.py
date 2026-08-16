"""Tests for IPT and the SSJ->IPT hybrid. Run with pytest, or directly:
python3 tests/test_ipt.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssj import (ipt_eigh, ipt_eig, ipt_eig_partial, ipt_hybrid_eigh,  # noqa: E402
                 ipt_rate_columns, refine_eig, ssj_ipt_eigh)
from ssj.ipt import ipt_rate  # noqa: E402


def near_diagonal(n, ratio, seed=0):
    rng = np.random.default_rng(seed)
    d = np.arange(n, dtype=float)
    W = rng.standard_normal((n, n))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    W *= ratio / np.max(np.abs(W))
    return np.diag(d) + W


def goe(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M + M.T) / np.sqrt(2.0 * n)


def check(A, w, V, dlam_tol=1e-12, resid_tol=1e-11, ortho_tol=1e-10):
    n = A.shape[0]
    norm2 = np.linalg.norm(A, ord=2)
    assert np.all(np.diff(w) >= 0), "eigenvalues not sorted"
    assert np.max(np.abs(w - np.linalg.eigvalsh(A))) / norm2 < dlam_tol
    assert np.linalg.norm(A @ V - V * w, "fro") / norm2 < resid_tol
    assert np.linalg.norm(V.conj().T @ V - np.eye(n), "fro") < ortho_tol


def test_ipt_near_diagonal():
    A = near_diagonal(120, 0.01)
    w, V, info = ipt_eigh(A, return_info=True)
    assert info["converged"] and info["iters"] <= 10
    check(A, w, V)


def test_ipt_reports_divergence_outside_basin():
    # GOE is far outside IPT's basin: the failure must be reported, not hidden
    w, V, info = ipt_eigh(goe(80, seed=1), return_info=True)
    assert not info["converged"]


def test_ipt_rate_predicts_convergence():
    for ratio, expect in [(0.01, True), (5.0, False)]:
        A = near_diagonal(60, ratio)
        r = ipt_rate(A)
        _, _, info = ipt_eigh(A, return_info=True)
        assert (r < 0.5) == expect
        assert info["converged"] == expect


def test_ipt_rate_infinite_on_exact_ties():
    A = np.diag([1.0, 1.0, 2.0])
    A[0, 1] = A[1, 0] = 0.1
    assert not np.isfinite(ipt_rate(A))


def test_hybrid_global_on_goe():
    A = goe(120, seed=2)
    w, V, info = ssj_ipt_eigh(A, return_info=True)
    assert info["converged"]
    check(A, w, V)


def test_hybrid_takes_pure_ipt_path_when_near_diagonal():
    A = near_diagonal(100, 0.01)
    w, V, info = ssj_ipt_eigh(A, return_info=True)
    assert info["path"] == "ipt" and info["sweeps"] == 0
    check(A, w, V)


def test_hybrid_handles_exact_degeneracies():
    # clustered spectrum: the gate opens but IPT cannot resolve the cluster,
    # so the hybrid must fall back to SSJ rather than return a wrong answer
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((60, 60)))
    vals = np.repeat(rng.uniform(-1, 1, 12), 5)
    A = (Q * vals) @ Q.T
    A = (A + A.T) / 2.0
    w, V, info = ssj_ipt_eigh(A, return_info=True)
    assert info["converged"]
    check(A, w, V)


def test_hybrid_warm_start():
    A = goe(100, seed=4)
    _, V0 = np.linalg.eigh(A)
    rng = np.random.default_rng(5)
    P = rng.standard_normal((100, 100))
    P = (P + P.T) / 2.0
    A2 = A + 1e-5 * P / np.linalg.norm(P, 2)
    w, V, info = ssj_ipt_eigh(A2, X0=V0, return_info=True)
    assert info["converged"] and info["sweeps"] == 0
    check(A2, w, V)


def test_ipt_complex_hermitian():
    rng = np.random.default_rng(6)
    n = 60
    W = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    W = (W + W.conj().T) / 2.0
    np.fill_diagonal(W, 0.0)
    A = np.diag(np.arange(n, dtype=float)) + 0.01 * W / np.max(np.abs(W))
    w, V, info = ipt_eigh(A, return_info=True)
    assert info["converged"]
    check(A, w, V)


def test_ipt_general_nonsymmetric():
    """IPT on a general (nonsymmetric) near-diagonal matrix."""
    rng = np.random.default_rng(10)
    n = 80
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    A = np.diag(np.arange(n, dtype=float)) + 0.01 * W / np.max(np.abs(W))
    w, V, info = ipt_eig(A, return_info=True)
    assert info["converged"] and info["iters"] <= 8
    norm2 = np.linalg.norm(A, ord=2)
    # eigenvalues match LAPACK (spectrum is real in this regime)
    assert np.max(np.abs(np.sort(w.real) - np.sort(np.linalg.eigvals(A).real))) \
        / norm2 < 1e-12
    assert np.max(np.abs(w.imag)) / norm2 < 1e-12
    # residual, with NO orthogonality expected of V
    assert np.linalg.norm(A @ V - V * w, "fro") / norm2 < 1e-11


def test_ipt_general_does_not_orthogonalize():
    """Eigenvectors of a nonsymmetric matrix are not orthogonal; the solver
    must not force them to be (that would be a wrong answer, not a slow one)."""
    rng = np.random.default_rng(11)
    n = 60
    W = rng.standard_normal((n, n))
    np.fill_diagonal(W, 0.0)
    A = np.diag(np.arange(n, dtype=float)) + 0.2 * W / np.max(np.abs(W))
    w, V, info = ipt_eig(A, return_info=True)
    assert info["converged"]
    off_orth = np.linalg.norm(V.conj().T @ V - np.eye(n), "fro")
    assert off_orth > 1e-6, "V came back orthogonal; symmetry was assumed somewhere"
    assert np.linalg.norm(A @ V - V * w, "fro") / np.linalg.norm(A, 2) < 1e-11


def test_ipt_general_reports_divergence():
    rng = np.random.default_rng(12)
    A = rng.standard_normal((60, 60))  # Ginibre: far outside the basin
    _, _, info = ipt_eig(A, return_info=True)
    assert not info["converged"]


def _match_err(a, b, scale):
    b = list(b)
    worst = 0.0
    for x in a:
        d = [abs(x - y) for y in b]
        k = int(np.argmin(d))
        worst = max(worst, d[k])
        b.pop(k)
    return worst / scale


def test_refine_eig_lifts_float32_to_double():
    """IPT as a refinement engine: a float32 LAPACK solve is ~1e-8 accurate;
    a few IPT iterations must take it to full double precision. This works on
    a DENSE GINIBRE matrix -- far outside IPT's own near-diagonal basin --
    because the presolve supplies the frame."""
    rng = np.random.default_rng(5)
    n = 200
    A = rng.standard_normal((n, n))
    scale = np.linalg.norm(A, 2)
    exact = np.linalg.eigvals(A)

    w0, V0 = np.linalg.eig(A.astype(np.float32))
    err0 = _match_err(w0.astype(complex), exact, scale)
    assert err0 > 1e-10, "float32 presolve should NOT already be exact"

    w, V, info = refine_eig(A, w0, V0, return_info=True)
    assert info["converged"] and info["iters"] <= 6
    assert info["rate"] < 1e-3          # the presolve lands deep in the basin
    assert _match_err(w, exact, scale) < 1e-12
    assert np.linalg.norm(A @ V - V * w, "fro") / scale < 1e-11
    # improvement of at least four orders of magnitude
    assert _match_err(w, exact, scale) < err0 / 1e4


def test_refine_eig_reports_a_bad_presolve():
    """A useless presolve must be reported, not silently returned as refined."""
    rng = np.random.default_rng(6)
    n = 60
    A = rng.standard_normal((n, n))
    junk = rng.standard_normal((n, n))            # not an eigenbasis at all
    w, V, info = refine_eig(A, np.diag(junk), junk, return_info=True)
    assert info["rate"] > 1.0 and not info["converged"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


def test_hybrid_composes_with_block_preconditioner():
    """ssj_ipt_eigh(block_m=32): the globalizer (BC) and the manifold-free
    endgame (IPT) compose. This exact combination once returned 1e-8
    eigenvalue error, because the coarse globalizing blocks left the frame
    orthonormal only to their own loose tolerance and IPT inherited the
    defect as a similarity error. The hand-off now re-orthonormalizes, and
    this pins full precision plus the fact that the hand-off actually fires.
    """
    import numpy as np
    from ssj import ssj_ipt_eigh

    r = np.random.default_rng(1)
    M = r.standard_normal((400, 400))
    A = (M + M.T) / np.sqrt(800)
    w, V, info = ssj_ipt_eigh(A, block_m=32, return_info=True)
    nrm = np.linalg.norm(A, 2)
    assert info["path"] == "hybrid"          # the gate opened
    assert info["sweeps"] <= 12              # BC did the globalizing
    assert np.max(np.abs(np.sort(w) - np.linalg.eigvalsh(A))) / nrm < 1e-12
    assert np.max(np.linalg.norm(A @ V - V * w, axis=0)) / nrm < 1e-11


def test_hybrid_falls_back_to_ssj_on_exact_ties():
    """Exact degeneracy: rho is infinite, the gate must never open, and the
    hybrid must finish as pure SSJ at full accuracy."""
    import numpy as np
    from ssj import ssj_ipt_eigh

    r = np.random.default_rng(2)
    Q, _ = np.linalg.qr(r.standard_normal((200, 200)))
    vals = np.repeat(r.standard_normal(41), 5)[:200]
    A = (Q * vals) @ Q.T
    A = (A + A.T) / 2
    w, V, info = ssj_ipt_eigh(A, block_m=32, return_info=True)
    nrm = np.linalg.norm(A, 2)
    assert info["path"] == "ssj"
    assert np.max(np.abs(np.sort(w) - np.linalg.eigvalsh(A))) / nrm < 1e-12


# --------------------------------------------------------------------------
# ipt_rate_columns fast path, and the column-split hybrid.
# --------------------------------------------------------------------------

def _clustered(n, k, eps=3e-4, gap=1e-7, seed=0):
    """Well-separated diagonal carrying one tight k-cluster: the global IPT
    rate is huge, but only ~k columns are actually resonant."""
    r = np.random.default_rng(seed)
    d = np.arange(n, dtype=float)
    d[n // 2:n // 2 + k] = d[n // 2] + gap * np.arange(k)
    W = r.standard_normal((n, n))
    W = (W + W.T) / 2
    return np.diag(d) + eps * W


def _rate_columns_reference(A, cols):
    """The per-column loop the vectorized path replaced, kept verbatim as an
    oracle. The shipped fast path must agree with it BIT for bit -- it is not
    an approximation, it is the same expression evaluated in blocks."""
    A = np.asarray(A)
    d = np.diag(A)
    out = np.empty(len(cols))
    for j, c in enumerate(cols):
        gap = np.abs(d[c] - d)
        w = np.abs(A[:, c])
        mask = np.arange(A.shape[0]) != c
        g = gap[mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(g > 0, w[mask] / np.where(g > 0, g, 1.0), np.inf)
        out[j] = float(np.max(r))
    return out


@pytest.mark.parametrize("n", [7, 64, 200])
def test_rate_columns_fast_path_is_bit_identical(n):
    rng = np.random.default_rng(11)
    A = rng.standard_normal((n, n))
    A = (A + A.T) / np.sqrt(2 * n)
    for cols in (np.arange(n), np.arange(n)[::7], np.array([0]),
                 np.array([n - 1, 0, n // 2])):
        got = ipt_rate_columns(A, cols)
        want = _rate_columns_reference(A, cols)
        assert np.array_equal(got, want), (n, len(cols))


def test_rate_columns_fast_path_handles_exact_degeneracy():
    """gap == 0 must give +inf (divergent), and 0/0 must too -- the blocked
    form gets there through IEEE rather than np.where, so pin the behaviour."""
    A = np.diag([1.0, 1.0, 5.0])
    A[0, 1] = A[1, 0] = 0.3            # degenerate pair, coupled
    r = ipt_rate_columns(A, np.arange(3))
    assert np.isinf(r[0]) and np.isinf(r[1])
    assert np.isfinite(r[2])
    assert np.array_equal(r, _rate_columns_reference(A, np.arange(3)))

    Z = np.diag([2.0, 2.0, 9.0])       # degenerate pair, UNcoupled: 0/0
    r = ipt_rate_columns(Z, np.arange(3))
    assert np.array_equal(r, _rate_columns_reference(Z, np.arange(3)))


def test_rate_columns_empty():
    A = np.diag([1.0, 2.0, 3.0])
    assert ipt_rate_columns(A, np.array([], dtype=int)).shape == (0,)


@pytest.mark.parametrize("n,k", [(200, 5), (300, 20)])
def test_hybrid_solves_what_global_ipt_cannot(n, k):
    """The whole point: a tight cluster disqualifies IPT globally, but only k
    columns are actually resonant, and the map is column-separable."""
    A = _clustered(n, k, seed=1)
    assert ipt_rate(A) > 1.0, "test matrix is not outside the global basin"

    w, V, info = ipt_hybrid_eigh(A, return_info=True)
    nrm = np.linalg.norm(A, 2)
    wref = np.linalg.eigvalsh(A)
    assert np.max(np.abs(w - wref)) / nrm < 1e-12
    assert np.max(np.linalg.norm(A @ V - V * w, axis=0)) / nrm < 1e-11
    assert np.linalg.norm(V.T @ V - np.eye(n)) < 1e-9
    assert info["n_dense"] >= k          # the cluster went to the dense block
    assert info["n_ipt"] > n - 4 * k     # and almost everything else did not


def test_hybrid_matches_plain_ipt_when_the_whole_matrix_is_admissible():
    """With no cluster, every column passes the screen and the hybrid must
    degenerate to plain IPT rather than paying for a dense block."""
    rng = np.random.default_rng(4)
    n = 120
    A = np.diag(np.arange(n, dtype=float))
    W = rng.standard_normal((n, n))
    A = A + 1e-3 * (W + W.T) / 2
    w, V, info = ipt_hybrid_eigh(A, return_info=True)
    assert info["n_dense"] == 0
    wref = np.linalg.eigvalsh(A)
    assert np.max(np.abs(w - wref)) / np.linalg.norm(A, 2) < 1e-12


def test_hybrid_falls_through_to_dense_when_nothing_is_separated():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((60, 60))
    A = (A + A.T) / 2                    # dense GOE: no column is separated
    w, V, info = ipt_hybrid_eigh(A, return_info=True)
    assert info["path"] == "dense"
    assert np.allclose(w, np.linalg.eigvalsh(A))


def test_per_column_convergence_is_not_a_complete_basis():
    """Column separability makes the partial solve exact -- and is also why two
    columns can converge to the SAME eigenpair with nothing to stop them. Both
    report tiny step norms, because per-column convergence says nothing about
    the basis. This is why the screen cannot be replaced by the solver's own
    `converged` flag; the hybrid must stay accurate where the unscreened run
    silently loses rank.
    """
    n, k = 1600, 20
    A = _clustered(n, k, seed=1)

    # unscreened: IPT on every column, trusting only its own convergence flag
    w_all, V_all, info = ipt_eig_partial(A, np.arange(n), tol=1e-13,
                                         hermitian=True, return_info=True)
    conv = info["converged_cols"]
    Vc = V_all[:, conv]
    deficiency = int(conv.sum()) - int(np.linalg.matrix_rank(Vc, tol=1e-8))
    assert deficiency > 0, (
        "expected at least one pair of converged columns to collapse onto the "
        "same eigenpair; if this stops happening the screen's rationale needs "
        "re-measuring, not the assertion relaxing")

    # the shipped hybrid screens first, and is unaffected
    w, V, hinfo = ipt_hybrid_eigh(A, return_info=True)
    nrm = np.linalg.norm(A, 2)
    assert np.max(np.abs(w - np.linalg.eigvalsh(A))) / nrm < 1e-12
    assert np.linalg.norm(V.T @ V - np.eye(n)) < 1e-9
