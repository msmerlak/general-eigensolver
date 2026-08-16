/* Spectral divide and conquer in C: matrix sign function + one orthogonal
 * split, leaves to dgeev.
 *
 * A compiled port of ssj.sdc (SSJ_LOG #24-25), written to settle one
 * specific contradiction. The Python SDC has *operation-count parity* with
 * dgeev on Ginibre n=400 -- 88 gemm-equivalents counted against dgeev's 89 --
 * and still loses 7.7-14x at the wall, with the entire discrepancy localized
 * inside `matrix_sign`. Either that gap is NumPy substrate, exactly as the
 * symmetric side's was (#13/#23), or the operation model is wrong. This
 * decides it, and instruments the phases so the answer is attributed rather
 * than inferred.
 *
 * The algorithm, unchanged from the Python:
 *   1. scale X = (A - sigma I)/||A - sigma I||_2, the norm by power iteration
 *      (O(n^2) and tight; the cheap sqrt(||.||_1 ||.||_inf) bound over-scales
 *      a random matrix badly and ADDS Newton steps)
 *   2. sign iteration: scaled Newton X <- (mu X + mu^-1 X^-1)/2 while far,
 *      Newton-Schulz X <- X(3I - X^2)/2 once ||X^2 - I||/sqrt(n) < 0.6
 *   3. P = (I + S)/2, r = round(trace P); pivoted QR of P gives an orthonormal
 *      basis whose first r columns span the invariant subspace
 *   4. B = Q^T A Q is block upper triangular; recurse, leaves to dgeev
 *
 * What C buys over NumPy here, in the order the flop model ranks it:
 *   - dgetri for the inverse instead of lu_solve against a full identity.
 *     The Python builds an n x n identity and runs n triangular solves
 *     (2n^3/3 + 2n^3 flops); dgetri is 4n^3/3 and never forms the identity.
 *   - ||X^2 - I||_F computed in one pass over X2, never materializing the
 *     difference. NumPy allocates and traverses an n x n temporary per
 *     iteration, and this norm is evaluated EVERY iteration.
 *   - the Newton combination (mu X + Xi/mu)/2 as one fused pass instead of
 *     NumPy's four (scale, scale, add, scale).
 *   - the Newton-Schulz target 1.5I - 0.5 X^2 built in place over X2, so the
 *     endgame is one gemm plus one pass, not one gemm plus three temporaries.
 *   - no allocation anywhere in the iteration; the sign loop ping-pongs two
 *     preallocated buffers.
 *
 * Column-major throughout. Symbols are scipy_*_64_ so this links the SAME
 * 64-bit-int OpenBLAS NumPy uses -- #13's cross-library confound, where
 * SciPy's BLAS wrapper measured 2.6x slower than NumPy's `@` on this box and
 * nearly produced a false verdict, is not repeated here.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sdc_eig.h"

typedef int64_t bi;

extern void scipy_dgemm_64_(const char *, const char *, const bi *, const bi *,
                            const bi *, const double *, const double *,
                            const bi *, const double *, const bi *,
                            const double *, double *, const bi *);
extern void scipy_dgemv_64_(const char *, const bi *, const bi *,
                            const double *, const double *, const bi *,
                            const double *, const bi *, const double *,
                            double *, const bi *);
extern void scipy_dgetrf_64_(const bi *, const bi *, double *, const bi *,
                             bi *, bi *);
extern void scipy_dgetri_64_(const bi *, double *, const bi *, const bi *,
                             double *, const bi *, bi *);
extern void scipy_dgeqp3_64_(const bi *, const bi *, double *, const bi *,
                             bi *, double *, double *, const bi *, bi *);
extern void scipy_dorgqr_64_(const bi *, const bi *, const bi *, double *,
                             const bi *, const double *, double *, const bi *,
                             bi *);
extern void scipy_dgeev_64_(const char *, const char *, const bi *, double *,
                            const bi *, double *, double *, double *,
                            const bi *, double *, const bi *, double *,
                            const bi *, bi *);

#define IDX(i, j, ld) ((size_t)(i) + (size_t)(ld) * (size_t)(j))

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static double *xmalloc_d(size_t k) {
    double *p = (double *)malloc(k * sizeof(double));
    if (!p) { fprintf(stderr, "OOM %zu doubles\n", k); exit(1); }
    return p;
}

/* Handoff threshold from scaled Newton to Newton-Schulz, on
 * ||X^2 - I||_F/sqrt(n). Newton-Schulz for the sign function converges only
 * inside ||I - X^2|| < 1, so 1.0 is a hard ceiling in the operator norm; this
 * measure is normalized by sqrt(n) and is therefore not that norm, which is
 * exactly why the useful value is measured rather than derived.
 *
 * 0.9 measured best at n=200 and n=400 and within noise of best at n=800,
 * and is never worse than the previous 0.6 anywhere in a sweep over
 * {0.6, 0.8, 0.9, 0.95, 1.0, 1.1, 1.3, 1.6}. But the honest size of the win
 * is 3-5%, which overlaps this box's contamination band -- record it as
 * noise-level, not as a result.
 *
 * THIS PARAMETER IS NOT A LEVER, and that is the real finding (SSJ_LOG #29).
 * The iteration converges quadratically, so dev crosses the entire disputed
 * band in one or two steps: of 16 steps at n=800, exactly 2 land in
 * [0.6, 1.6]. There is almost nothing there to reassign no matter where the
 * threshold sits. Past 1.0 it gets monotonically WORSE (0.66x -> 0.60x at
 * n=800) because NS entered outside its region converges slowly and the step
 * count grows 6 -> 12. The sign iteration's cost is set by the 8-11 steps
 * spent FAR from convergence, where only Newton works. */
static double g_ns_switch = 0.9;
void sdc_set_ns_switch(double v) { g_ns_switch = v; }
double sdc_get_ns_switch(void) { return g_ns_switch; }

/* Deterministic xorshift, so shift retries are reproducible run to run. */
static uint64_t g_rs = 0x5D1ULL;
static double rnd_normal(void) {
    uint64_t r = g_rs;
    r ^= r << 13; r ^= r >> 7; r ^= r << 17; g_rs = r;
    double u1 = ((double)((r >> 11) & ((1ULL << 53) - 1))) / 9007199254740992.0;
    r ^= r << 13; r ^= r >> 7; r ^= r << 17; g_rs = r;
    double u2 = ((double)((r >> 11) & ((1ULL << 53) - 1))) / 9007199254740992.0;
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* ------------------------------------------------------- scratch workspace */

/* One allocation for a whole solve. The sign iteration must not allocate:
 * at n=400 a malloc/free pair per iteration across ~15 iterations and 12
 * possible shift retries is pure noise on top of the thing being measured. */
struct sdc_ws {
    bi n;
    double *cur, *alt, *x2;   /* sign iteration buffers            */
    double *lu;               /* LU scratch (dgetrf destroys input) */
    double *q, *tmp, *b;      /* split: basis, A*Q, Q^T A Q         */
    double *v, *w;            /* power iteration vectors            */
    double *tau, *work;
    bi *ipiv, *jpvt;
    bi lwork;
};

static void ws_init(struct sdc_ws *ws, bi n) {
    size_t nn = (size_t)n * (size_t)n;
    ws->n = n;
    ws->cur = xmalloc_d(nn); ws->alt = xmalloc_d(nn); ws->x2 = xmalloc_d(nn);
    ws->lu = xmalloc_d(nn);
    ws->q = xmalloc_d(nn); ws->tmp = xmalloc_d(nn); ws->b = xmalloc_d(nn);
    ws->v = xmalloc_d((size_t)n); ws->w = xmalloc_d((size_t)n);
    ws->tau = xmalloc_d((size_t)n);
    ws->ipiv = (bi *)calloc((size_t)n, sizeof(bi));
    ws->jpvt = (bi *)calloc((size_t)n, sizeof(bi));

    /* Workspace query across every routine that needs one; take the max. */
    double q1 = 0, q2 = 0, q3 = 0, q4 = 0;
    bi mone = -1, info = 0;
    scipy_dgetri_64_(&n, ws->lu, &n, ws->ipiv, &q1, &mone, &info);
    scipy_dgeqp3_64_(&n, &n, ws->q, &n, ws->jpvt, ws->tau, &q2, &mone, &info);
    scipy_dorgqr_64_(&n, &n, &n, ws->q, &n, ws->tau, &q3, &mone, &info);
    scipy_dgeev_64_("N", "N", &n, ws->b, &n, ws->v, ws->w, NULL, &n, NULL, &n,
                    &q4, &mone, &info);
    double mx = q1 > q2 ? q1 : q2;
    if (q3 > mx) mx = q3;
    if (q4 > mx) mx = q4;
    ws->lwork = (bi)mx;
    if (ws->lwork < 4 * n + 64) ws->lwork = 4 * n + 64;
    ws->work = xmalloc_d((size_t)ws->lwork);
}

static void ws_free(struct sdc_ws *ws) {
    free(ws->cur); free(ws->alt); free(ws->x2); free(ws->lu);
    free(ws->q); free(ws->tmp); free(ws->b);
    free(ws->v); free(ws->w); free(ws->tau); free(ws->work);
    free(ws->ipiv); free(ws->jpvt);
}

/* ------------------------------------------------------------ small pieces */

/* ||X||_2 by power iteration on X^T X. O(n^2) per step and tight within a
 * few percent -- an SVD costs more than several Newton steps, and the cheap
 * sqrt(||X||_1 ||X||_inf) bound over-scales and adds them. */
static double spectral_norm(const double *X, bi n, double *v, double *w,
                            int iters) {
    const double one = 1.0, zero = 0.0;
    const bi i1 = 1;
    double s = 0.0;
    for (bi i = 0; i < n; i++) v[i] = 1.0 + 0.01 * (double)i;
    for (bi i = 0; i < n; i++) s += v[i] * v[i];
    s = 1.0 / sqrt(s);
    for (bi i = 0; i < n; i++) v[i] *= s;

    double est = 0.0;
    for (int k = 0; k < iters; k++) {
        scipy_dgemv_64_("N", &n, &n, &one, X, &n, v, &i1, &zero, w, &i1);
        scipy_dgemv_64_("T", &n, &n, &one, X, &n, w, &i1, &zero, v, &i1);
        est = 0.0;
        for (bi i = 0; i < n; i++) est += v[i] * v[i];
        est = sqrt(est);
        if (est == 0.0) return 0.0;
        double inv = 1.0 / est;
        for (bi i = 0; i < n; i++) v[i] *= inv;
    }
    return sqrt(est);
}

/* ||M - I||_F in ONE pass, without materializing the difference. NumPy pays
 * an n x n temporary here on every single iteration. */
static double dev_from_identity(const double *M, bi n) {
    double s = 0.0;
    for (bi j = 0; j < n; j++)
        for (bi i = 0; i < n; i++) {
            double d = M[IDX(i, j, n)] - (i == j ? 1.0 : 0.0);
            s += d * d;
        }
    return sqrt(s);
}

/* ---------------------------------------------------------- matrix sign fn */

/* Sign of X (n x n, column-major, overwritten). Returns iterations, or -1 if
 * an iterate went singular -- which is not a bug but the caller's cue to try
 * a different shift (an eigenvalue sat on the splitting line). */
static int matrix_sign_c(double *X, bi n, double tol, double ns_switch,
                         int max_iter, struct sdc_ws *ws, sdc_stats *st) {
    const double one = 1.0, zero = 0.0;
    const double sqn = sqrt((double)n);
    size_t nn = (size_t)n * (size_t)n;
    bi info = 0;

    double nrm = spectral_norm(X, n, ws->v, ws->w, 20);
    if (nrm == 0.0) return 0;
    double inv_nrm = 1.0 / nrm;
    for (size_t k = 0; k < nn; k++) X[k] *= inv_nrm;

    /* Ping-pong so the Newton-Schulz gemm never aliases its own input. */
    double *cur = X, *alt = ws->alt, *x2 = ws->x2;

    int it = 0;
    int converged = 0;
    for (it = 1; it <= max_iter; it++) {
        scipy_dgemm_64_("N", "N", &n, &n, &n, &one, cur, &n, cur, &n, &zero,
                        x2, &n);
        double dev = dev_from_identity(x2, n) / sqn;
        if (st) st->last_dev = dev;
        if (dev < tol) { converged = 1; break; }

        if (dev < ns_switch) {
            /* Newton-Schulz: 2 gemms, no factorization, quadratic.
             * Build 1.5I - 0.5 X^2 in place over X2, then one gemm. */
            for (size_t k = 0; k < nn; k++) x2[k] *= -0.5;
            for (bi i = 0; i < n; i++) x2[IDX(i, i, n)] += 1.5;
            scipy_dgemm_64_("N", "N", &n, &n, &n, &one, cur, &n, x2, &n, &zero,
                            alt, &n);
            double *t = cur; cur = alt; alt = t;
            if (st) st->n_ns++;
        } else {
            /* Scaled Newton. One LU serves both the determinant and the
             * inverse -- a separate slogdet would factorize twice. */
            memcpy(ws->lu, cur, nn * sizeof(double));
            scipy_dgetrf_64_(&n, &n, ws->lu, &n, ws->ipiv, &info);
            if (info != 0) return -1;              /* singular iterate */
            double logabsdet = 0.0;
            for (bi i = 0; i < n; i++)
                logabsdet += log(fabs(ws->lu[IDX(i, i, n)]));
            if (!isfinite(logabsdet)) return -1;
            scipy_dgetri_64_(&n, ws->lu, &n, ws->ipiv, ws->work, &ws->lwork,
                             &info);
            if (info != 0) return -1;
            double mu = exp(-logabsdet / (double)n);
            double a = 0.5 * mu, b = 0.5 / mu;
            for (size_t k = 0; k < nn; k++)        /* one fused pass */
                cur[k] = a * cur[k] + b * ws->lu[k];
            if (st) st->n_newton++;
        }
    }
    if (it > max_iter) it = max_iter;
    if (cur != X) memcpy(X, cur, nn * sizeof(double));
    if (!converged && st) st->n_fail_maxiter++;
    return converged ? it : -2 - it;   /* negative encodes non-convergence */
}

/* --------------------------------------------------------------- one split */

/* Split A at Re(z) = shift. On success fills ws->b with Q^T A Q and returns
 * the rank r; returns -1 for a degenerate split, -2 for a singular iterate,
 * -3 when the (2,1) block is too large to trust. */
static bi split_once(const double *A, bi n, double shift, double tol,
                     double normA, struct sdc_ws *ws, sdc_stats *st) {
    const double one = 1.0, zero = 0.0;
    size_t nn = (size_t)n * (size_t)n;
    bi info = 0;

    memcpy(ws->cur, A, nn * sizeof(double));
    for (bi i = 0; i < n; i++) ws->cur[IDX(i, i, n)] -= shift;

    double t0 = now_s();
    int its = matrix_sign_c(ws->cur, n, tol, g_ns_switch, 60, ws, st);
    if (st) { st->t_sign += now_s() - t0; st->n_sign_calls++; }
    if (its == -1) { if (st) { st->n_fail_singular++; } return -2; }
    if (its < 0) {                       /* ran out of iterations */
        if (st) st->iters_wasted += -(its + 2);
        return -4;
    }
    if (st) st->last_sign_iters = its;

    /* P = (I + S)/2, in place. */
    for (size_t k = 0; k < nn; k++) ws->cur[k] *= 0.5;
    double tr = 0.0;
    for (bi i = 0; i < n; i++) { ws->cur[IDX(i, i, n)] += 0.5;
                                 tr += ws->cur[IDX(i, i, n)]; }
    bi r = (bi)llround(tr);
    if (r <= 0 || r >= n) {
        if (st) { st->n_fail_rank++; st->iters_wasted += its; }
        return -1;
    }

    /* Orthonormal basis for range(P) by pivoted QR: the r independent columns
     * come first, so the leading r columns of Q span the invariant subspace.
     * (SSJ_LOG #24: building the basis from QR of [P, I-P] instead is WRONG --
     * the column reordering destroys the range separation.) */
    t0 = now_s();
    memcpy(ws->q, ws->cur, nn * sizeof(double));
    for (bi j = 0; j < n; j++) ws->jpvt[j] = 0;          /* all free to pivot */
    scipy_dgeqp3_64_(&n, &n, ws->q, &n, ws->jpvt, ws->tau, ws->work,
                     &ws->lwork, &info);
    if (info != 0) return -2;
    scipy_dorgqr_64_(&n, &n, &n, ws->q, &n, ws->tau, ws->work, &ws->lwork,
                     &info);
    if (info != 0) return -2;
    if (st) st->t_qr += now_s() - t0;

    /* B = Q^T A Q, two gemms. */
    t0 = now_s();
    scipy_dgemm_64_("N", "N", &n, &n, &n, &one, A, &n, ws->q, &n, &zero,
                    ws->tmp, &n);
    scipy_dgemm_64_("T", "N", &n, &n, &n, &one, ws->q, &n, ws->tmp, &n, &zero,
                    ws->b, &n);
    if (st) st->t_gemm += now_s() - t0;

    /* The (2,1) block is zero in exact arithmetic; how far it misses IS the
     * split's backward error, so it is checked rather than assumed. */
    double s = 0.0;
    for (bi j = 0; j < r; j++)
        for (bi i = r; i < n; i++) {
            double v = ws->b[IDX(i, j, n)]; s += v * v;
        }
    if (sqrt(s) / (normA > 0 ? normA : 1.0) > 1e-6) {
        if (st) { st->n_fail_resid++; st->iters_wasted += its; }
        return -3;
    }
    return r;
}

/* ------------------------------------------------------------- the recursion */

static void leaf_eigvals(double *M, bi n, bi ld, double *wr, double *wi,
                         struct sdc_ws *ws, sdc_stats *st) {
    bi info = 0;
    double t0 = now_s();
    if (n == 1) {
        wr[0] = M[0]; wi[0] = 0.0;
    } else if (n == 2) {
        double a = M[IDX(0, 0, ld)], b = M[IDX(0, 1, ld)];
        double c = M[IDX(1, 0, ld)], d = M[IDX(1, 1, ld)];
        double tr = a + d, det = a * d - b * c;
        double disc = tr * tr / 4.0 - det;
        if (disc >= 0.0) {
            double rt = sqrt(disc);
            wr[0] = tr / 2.0 + rt; wi[0] = 0.0;
            wr[1] = tr / 2.0 - rt; wi[1] = 0.0;
        } else {
            double rt = sqrt(-disc);
            wr[0] = tr / 2.0; wi[0] = rt;
            wr[1] = tr / 2.0; wi[1] = -rt;
        }
    } else {
        /* dgeev destroys its input, so hand it a copy it may consume. */
        double *C = xmalloc_d((size_t)n * (size_t)n);
        for (bi j = 0; j < n; j++)
            for (bi i = 0; i < n; i++) C[IDX(i, j, n)] = M[IDX(i, j, ld)];
        scipy_dgeev_64_("N", "N", &n, C, &n, wr, wi, NULL, &n, NULL, &n,
                        ws->work, &ws->lwork, &info);
        free(C);
    }
    if (st) st->t_leaf += now_s() - t0;
}

static void sdc_rec(const double *A, bi n, bi lda, double *wr, double *wi,
                    bi min_block, int depth, double tol, sdc_stats *st) {
    if (n <= min_block || depth >= 32) {
        struct sdc_ws lws;
        ws_init(&lws, n < 2 ? 2 : n);
        leaf_eigvals((double *)A, n, lda, wr, wi, &lws, st);
        ws_free(&lws);
        return;
    }

    /* Pack to a contiguous n x n block: the split does several full-size
     * gemms and factorizations on it, so a strided view would cost more
     * than the copy. */
    size_t nn = (size_t)n * (size_t)n;
    double *M = xmalloc_d(nn);
    for (bi j = 0; j < n; j++)
        for (bi i = 0; i < n; i++) M[IDX(i, j, n)] = A[IDX(i, j, lda)];

    struct sdc_ws ws;
    ws_init(&ws, n);

    double normA = 0.0, tr = 0.0;
    for (size_t k = 0; k < nn; k++) normA += M[k] * M[k];
    normA = sqrt(normA);
    for (bi i = 0; i < n; i++) tr += M[IDX(i, i, n)];
    double centre = tr / (double)n;
    double spread = normA / sqrt((double)n);

    bi r = -1;
    for (int attempt = 0; attempt < 12; attempt++) {
        double shift = (attempt == 0)
                     ? centre
                     : centre + spread * rnd_normal()
                                * pow(0.5, (double)(attempt / 4));
        r = split_once(M, n, shift, tol, normA, &ws, st);
        if (r > 0) break;
    }

    if (r <= 0) {
        /* No usable split (e.g. a genuine multiple eigenvalue filling the
         * block): fall back rather than loop or return something wrong. */
        leaf_eigvals(M, n, n, wr, wi, &ws, st);
        if (st) st->n_fallbacks++;
        ws_free(&ws); free(M);
        return;
    }

    /* ws.b holds Q^T A Q. Copy the two diagonal blocks out before recursing:
     * the child solves reuse their own workspace and would clobber it. */
    bi n2 = n - r;
    double *B11 = xmalloc_d((size_t)r * (size_t)r);
    double *B22 = xmalloc_d((size_t)n2 * (size_t)n2);
    for (bi j = 0; j < r; j++)
        for (bi i = 0; i < r; i++) B11[IDX(i, j, r)] = ws.b[IDX(i, j, n)];
    for (bi j = 0; j < n2; j++)
        for (bi i = 0; i < n2; i++)
            B22[IDX(i, j, n2)] = ws.b[IDX(r + i, r + j, n)];
    ws_free(&ws); free(M);

    sdc_rec(B11, r, r, wr, wi, min_block, depth + 1, tol, st);
    sdc_rec(B22, n2, n2, wr + r, wi + r, min_block, depth + 1, tol, st);
    free(B11); free(B22);
}

/* --------------------------------------------------------------- public API */

void sdc_eigvals_c(const double *A, bi n, double *wr, double *wi, bi min_block,
                   sdc_stats *st) {
    if (st) memset(st, 0, sizeof(*st));
    g_rs = 0x5D1ULL;                          /* reproducible shift retries */
    if (min_block <= 0) {
        /* 3n/5, NOT n/2. The centred split returns r = trace(P), which lands
         * NEAR n/2 but essentially never ON it, so with a leaf of exactly n/2
         * one half is a few rows too big and buys a whole second sign
         * iteration -- ~1/3 of the run -- to shave a dgeev that was already
         * cheap. Measured on Ginibre: 2 sign calls at 0.5n against 1 at every
         * fraction from 0.55n to 0.9n, worth 1.10x-1.16x with identical
         * accuracy, and flat across that whole range so the constant is not
         * delicate. Above 0.5n but well below 1 keeps the other property that
         * matters: a genuinely LOPSIDED split still leaves a big block that
         * recurses. Third appearance of the leaf lesson (#17, #25); this time
         * the wrong leaf was not "too deep" but "off by three". */
        min_block = (3 * n) / 5;
        if (min_block < 2) min_block = 2;
    }
    sdc_rec(A, n, n, wr, wi, min_block, 0, 1e-12, st);
}

void lapack_eig_c(const double *A, bi n, double *wr, double *wi) {
    bi info = 0, lwork = -1;
    double qw = 0.0;
    size_t nn = (size_t)n * (size_t)n;
    double *C = xmalloc_d(nn);
    memcpy(C, A, nn * sizeof(double));
    scipy_dgeev_64_("N", "N", &n, C, &n, wr, wi, NULL, &n, NULL, &n, &qw,
                    &lwork, &info);
    lwork = (bi)qw;
    if (lwork < 4 * n) lwork = 4 * n;
    double *work = xmalloc_d((size_t)lwork);
    memcpy(C, A, nn * sizeof(double));
    scipy_dgeev_64_("N", "N", &n, C, &n, wr, wi, NULL, &n, NULL, &n, work,
                    &lwork, &info);
    free(work); free(C);
}

/* Isolated cost of one sign call, for attribution against the whole solve. */
int matrix_sign_bench(const double *A, bi n, double shift, sdc_stats *st) {
    if (st) memset(st, 0, sizeof(*st));   /* per-call, not accumulated */
    struct sdc_ws ws;
    ws_init(&ws, n);
    size_t nn = (size_t)n * (size_t)n;
    memcpy(ws.cur, A, nn * sizeof(double));
    for (bi i = 0; i < n; i++) ws.cur[IDX(i, i, n)] -= shift;
    int its = matrix_sign_c(ws.cur, n, 1e-12, g_ns_switch, 60, &ws, st);
    ws_free(&ws);
    return its;
}
