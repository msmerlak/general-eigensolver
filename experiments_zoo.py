"""A zoo of rewritings of Av = lambda v as fixed points. All get the same
matrix, the same target, the same tolerance."""
import numpy as np

TOL, MAXIT = 1e-13, 300

def _setup(A, j):
    d = np.diag(A).astype(float).copy()
    W = A - np.diag(d)
    return d, W

# 1. IPT: v_i = (Wv)_i/(lam - d_i), lam = d_j + (Wv)_j
def ipt(A, j):
    d, W = _setup(A, j); n = len(d)
    v = np.zeros(n); v[j] = 1.0; prev = np.inf
    for it in range(1, MAXIT+1):
        Wv = W @ v; lam = d[j] + Wv[j]
        den = lam - d; den[j] = 1.0
        u = Wv/den; u[j] = 1.0
        e = np.max(np.abs(u-v)); v = u
        if e <= TOL: return lam, v, it, True
        if not np.isfinite(e) or e > 1e3*prev: return lam, v, it, False
        prev = max(e,1e-300)
    return lam, v, MAXIT, False

# 2. IPT but lambda from the RAYLEIGH QUOTIENT (2nd-order accurate for symmetric)
def ipt_rq(A, j):
    d, W = _setup(A, j); n = len(d)
    v = np.zeros(n); v[j] = 1.0; prev = np.inf; lam = d[j]
    for it in range(1, MAXIT+1):
        Av = A @ v; lam = float(v @ Av)/float(v @ v)
        Wv = W @ v
        den = lam - d; den[j] = 1.0
        u = Wv/den; u[j] = 1.0
        e = np.max(np.abs(u-v)); v = u
        if e <= TOL: return lam, v, it, True
        if not np.isfinite(e) or e > 1e3*prev: return lam, v, it, False
        prev = max(e,1e-300)
    return lam, v, MAXIT, False

# 3. Damped IPT: v <- v + beta (T(v)-v)
def ipt_damped(A, j, beta=0.5):
    d, W = _setup(A, j); n = len(d)
    v = np.zeros(n); v[j] = 1.0; prev = np.inf; lam = d[j]
    for it in range(1, MAXIT+1):
        Wv = W @ v; lam = d[j] + Wv[j]
        den = lam - d; den[j] = 1.0
        u = Wv/den; u[j] = 1.0
        u = v + beta*(u-v); u[j] = 1.0
        e = np.max(np.abs(u-v)); v = u
        if e <= TOL: return lam, v, it, True
        if not np.isfinite(e) or e > 1e3*prev: return lam, v, it, False
        prev = max(e,1e-300)
    return lam, v, MAXIT, False

# 4. Aitken/Shanks extrapolation on the iterate sequence (series resummation,
#    NOT iterate mixing: acts elementwise on three successive partial sums)
def ipt_aitken(A, j):
    d, W = _setup(A, j); n = len(d)
    v = np.zeros(n); v[j] = 1.0; hist = []; prev = np.inf; lam = d[j]
    for it in range(1, MAXIT+1):
        Wv = W @ v; lam = d[j] + Wv[j]
        den = lam - d; den[j] = 1.0
        u = Wv/den; u[j] = 1.0
        hist.append(u.copy())
        if len(hist) >= 3:
            x0,x1,x2 = hist[-3],hist[-2],hist[-1]
            dd = x2-2*x1+x0
            with np.errstate(divide='ignore',invalid='ignore'):
                acc = np.where(np.abs(dd)>1e-300, x2-(x2-x1)**2/dd, x2)
            if np.all(np.isfinite(acc)): u = acc; u[j]=1.0
            hist = hist[-2:]
        e = np.max(np.abs(u-v)); v = u
        if e <= TOL: return lam, v, it, True
        if not np.isfinite(e) or e > 1e3*prev: return lam, v, it, False
        prev = max(e,1e-300)
    return lam, v, MAXIT, False

# 5. Scalar self-consistent 2nd-order self-energy: lam = d_j + sum |W_jk|^2/(lam-d_k)
def self_energy(A, j):
    d, W = _setup(A, j); n = len(d)
    mask = np.arange(n) != j
    num = (W[j,:]*W[:,j])[mask]; dk = d[mask]
    lam = d[j]; prev = np.inf
    for it in range(1, MAXIT+1):
        den = lam - dk
        if np.min(np.abs(den)) < 1e-14: return lam, None, it, False
        new = d[j] + float(np.sum(num/den))
        e = abs(new-lam)/max(abs(new),1e-300); lam = new
        if e <= TOL: return lam, None, it, True
        if not np.isfinite(e) or e > 1e3*prev: return lam, None, it, False
        prev = max(e,1e-300)
    return lam, None, MAXIT, False

# 6. Davidson: diagonal-preconditioned residual, subspace + Rayleigh-Ritz
def davidson(A, j, mmax=40):
    d, W = _setup(A, j); n = len(d)
    v = np.zeros(n); v[j] = 1.0
    V = v[:,None]/np.linalg.norm(v)
    lam = d[j]
    for it in range(1, MAXIT+1):
        AV = A @ V
        H = V.T @ AV
        w_, S = np.linalg.eigh((H+H.T)/2) if np.allclose(H,H.T) else np.linalg.eig(H)
        m = int(np.argmin(np.abs(w_-d[j])))
        lam = float(np.real(w_[m])); y = np.real(S[:,m])
        u = V @ y; r = A @ u - lam*u
        e = np.linalg.norm(r)
        if e <= TOL: return lam, u, it, True
        den = lam - d; den[np.abs(den)<1e-12] = 1e-12
        t = r/den                                    # diagonal preconditioner
        t = t - V @ (V.T @ t)                        # orthogonalize
        nt = np.linalg.norm(t)
        if nt < 1e-14: return lam, u, it, False
        t /= nt
        V = np.column_stack([V, t]) if V.shape[1] < mmax else np.column_stack([u/np.linalg.norm(u), t])
    return lam, u, MAXIT, False

# 7. Richardson / Jacobi on the singular system: v <- v - tau(Av - lam v)
def richardson(A, j, tau=None):
    d, W = _setup(A, j); n = len(d)
    v = np.zeros(n); v[j] = 1.0; v/=np.linalg.norm(v)
    if tau is None: tau = 1.0/np.max(np.abs(d))
    prev = np.inf; lam = d[j]
    for it in range(1, MAXIT+1):
        Av = A @ v; lam = float(v@Av)
        r = Av - lam*v
        u = v - tau*r; u /= np.linalg.norm(u)
        e = np.linalg.norm(u-v); v = u
        if np.linalg.norm(A@v - (v@(A@v))*v) <= TOL: return float(v@(A@v)), v, it, True
        if not np.isfinite(e) or e > 1e3*prev: return lam, v, it, False
        prev = max(e,1e-300)
    return lam, v, MAXIT, False

VARIANTS = [("IPT", ipt), ("IPT+RayleighQ", ipt_rq), ("IPT damped .5", ipt_damped),
            ("IPT+Aitken", ipt_aitken), ("self-energy(lam)", self_energy),
            ("Davidson", davidson), ("Richardson", richardson)]
