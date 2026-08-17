/* Accuracy battery + wall race for the C spectral divide-and-conquer solver.
 *
 * Both sides run through the SAME binary and the SAME OpenBLAS, so the
 * comparison isolates the algorithm and its implementation rather than the
 * BLAS (OPTIMIZATION_LOG #13's cross-library confound).
 *
 * Every timed configuration is accuracy-checked BEFORE it is timed: a routine
 * that fails fast looks fast (OPTIMIZATION_LOG #5/#7).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sdc_eig.h"

typedef int64_t bi;

#define IDX(i, j, ld) ((size_t)(i) + (size_t)(ld) * (size_t)(j))

extern void scipy_dgemm_64_(const char *, const char *, const bi *, const bi *,
                            const bi *, const double *, const double *,
                            const bi *, const double *, const bi *,
                            const double *, double *, const bi *);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static double loadavg(void) {
    FILE *f = fopen("/proc/loadavg", "r");
    double l = -1.0;
    if (f) { if (fscanf(f, "%lf", &l) != 1) l = -1.0; fclose(f); }
    return l;
}

static uint64_t rs = 88172645463325252ULL;
static double rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    double u1 = ((double)((rs >> 11) & ((1ULL << 53) - 1))) / 9007199254740992.0;
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    double u2 = ((double)((rs >> 11) & ((1ULL << 53) - 1))) / 9007199254740992.0;
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* Real Ginibre: entries N(0, 1/n). Roughly half the spectrum is complex. */
static void ginibre(double *A, bi n, uint64_t seed) {
    rs = seed ? seed : 1;
    double s = 1.0 / sqrt((double)n);
    for (size_t k = 0; k < (size_t)n * (size_t)n; k++) A[k] = rnd() * s;
}

/* A real Schur form built directly: upper triangular T with a planted real
 * spectrum, conjugated by an orthogonal Q. Nonsymmetric, real spectrum,
 * eigenvalues known exactly.
 *
 * `up` scales the strictly-upper entries and therefore the non-normality.
 * It must stay SMALL, and that is a measured constraint rather than taste:
 * at up = 0.3 the eigenvalues of this construction are so ill-conditioned
 * that DGEEV ITSELF misses the planted spectrum by 1.5e-01 to 2.0e-01, so
 * any solver "failure" measured there is the test matrix's, not the
 * algorithm's. At up = 0.005 dgeev recovers the planted values to
 * 1.7e-14 (n=200) .. 2.4e-13 (n=800) and the case is meaningful. The bench
 * prints the dgeev-vs-planted number so the case keeps validating itself. */
static void planted_real(double *A, bi n, double *vals, double up,
                         uint64_t seed) {
    rs = seed ? seed : 1;
    size_t nn = (size_t)n * (size_t)n;
    double *T = calloc(nn, sizeof(double));
    double *Q = malloc(nn * sizeof(double));
    double *W = malloc(nn * sizeof(double));
    for (bi j = 0; j < n; j++) {
        for (bi i = 0; i < j; i++) T[IDX(i, j, n)] = rnd() * up;
        T[IDX(j, j, n)] = vals[j];
    }
    for (size_t k = 0; k < nn; k++) Q[k] = rnd();
    for (bi j = 0; j < n; j++) {                  /* Gram-Schmidt, off path */
        for (bi k = 0; k < j; k++) {
            double d = 0.0;
            for (bi i = 0; i < n; i++) d += Q[IDX(i, k, n)] * Q[IDX(i, j, n)];
            for (bi i = 0; i < n; i++) Q[IDX(i, j, n)] -= d * Q[IDX(i, k, n)];
        }
        double nr = 0.0;
        for (bi i = 0; i < n; i++) nr += Q[IDX(i, j, n)] * Q[IDX(i, j, n)];
        nr = 1.0 / sqrt(nr);
        for (bi i = 0; i < n; i++) Q[IDX(i, j, n)] *= nr;
    }
    const double one = 1.0, zero = 0.0;
    scipy_dgemm_64_("N", "N", &n, &n, &n, &one, Q, &n, T, &n, &zero, W, &n);
    scipy_dgemm_64_("N", "T", &n, &n, &n, &one, W, &n, Q, &n, &zero, A, &n);
    free(T); free(Q); free(W);
}

static double norm2_est(const double *A, bi n) {   /* Frobenius/sqrt(n) proxy */
    double s = 0.0;
    for (size_t k = 0; k < (size_t)n * (size_t)n; k++) s += A[k] * A[k];
    return sqrt(s);
}

/* Max matched distance between two spectra, by greedy nearest neighbour.
 * NOT a lexicographic sort: a real matrix has exact conjugate pairs whose
 * real parts tie, so sorting reports errors of order 2|Im lambda| that are
 * not there -- that bug cost a full re-measurement this campaign. */
static double spec_err(const double *ar, const double *ai,
                       const double *br, const double *bi_, bi n, double nrm) {
    char *used = calloc((size_t)n, 1);
    double worst = 0.0;
    for (bi i = 0; i < n; i++) {
        double best = 1e300; bi bj = -1;
        for (bi j = 0; j < n; j++) {
            if (used[j]) continue;
            double dr = ar[i] - br[j], di = ai[i] - bi_[j];
            double d = sqrt(dr * dr + di * di);
            if (d < best) { best = d; bj = j; }
        }
        if (bj >= 0) used[bj] = 1;
        if (best > worst) worst = best;
    }
    free(used);
    return worst / (nrm > 0 ? nrm : 1.0);
}

int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 3;
    if (reps < 1) reps = 1;
    bi sizes[] = {200, 400, 800};
    int nsizes = 3;

    printf("SDC in C -- spectral divide and conquer vs dgeev, same OpenBLAS\n");
    printf("load at start: %.2f   reps=%d\n", loadavg(), reps);

    /* ------------------------------- accuracy (load-immune, run first) --- */
    printf("\n=== accuracy vs dgeev (nearest-match spectrum distance)\n");
    printf("%-28s %8s %12s %10s %8s\n", "case", "n", "dlam", "sign its",
           "leaf");
    for (int si = 0; si < nsizes; si++) {
        bi n = sizes[si];
        size_t nn = (size_t)n * (size_t)n;
        double *A = malloc(nn * sizeof(double));
        double *wr = malloc((size_t)n * sizeof(double));
        double *wi = malloc((size_t)n * sizeof(double));
        double *rr = malloc((size_t)n * sizeof(double));
        double *ri = malloc((size_t)n * sizeof(double));

        for (int cs = 0; cs < 2; cs++) {
            const char *name;
            if (cs == 0) { ginibre(A, n, 2); name = "Ginibre"; }
            else {
                double *v = malloc((size_t)n * sizeof(double));
                for (bi i = 0; i < n; i++) v[i] = -1.0 + 2.0 * (double)i / (double)n;
                planted_real(A, n, v, 0.005, 7);
                /* self-check: is this case even well conditioned enough to
                 * ask the question? dgeev against the planted truth. */
                {
                    double *zi = calloc((size_t)n, sizeof(double));
                    double nr = norm2_est(A, n) / sqrt((double)n);
                    lapack_eig_c(A, n, rr, ri);
                    double dg = spec_err(rr, ri, v, zi, n, nr);
                    if (dg > 1e-10)
                        printf("  (skipping planted n=%lld: dgeev itself is "
                               "%.1e from the planted spectrum)\n",
                               (long long)n, dg);
                    free(zi);
                }
                free(v);
                name = "planted real spectrum";
            }
            double nrm = norm2_est(A, n) / sqrt((double)n);
            sdc_stats st;
            lapack_eig_c(A, n, rr, ri);
            sdc_eigvals_c(A, n, wr, wi, 0, &st);
            double e = spec_err(wr, wi, rr, ri, n, nrm);
            printf("%-28s %8lld %12.1e %10d %8d%s\n", name, (long long)n, e,
                   st.n_newton + st.n_ns, st.n_sign_calls,
                   st.n_fallbacks ? "  (fallback)" : "");
        }
        free(A); free(wr); free(wi); free(rr); free(ri);
    }

    /* ------------------------------------------------- the wall race ----- */
    printf("\n=== wall race on Ginibre (interleaved min-of-%d, "
           "accuracy asserted first)\n", reps);
    for (int si = 0; si < nsizes; si++) {
        bi n = sizes[si];
        size_t nn = (size_t)n * (size_t)n;
        double *A = malloc(nn * sizeof(double));
        ginibre(A, n, 2);
        double *wr = malloc((size_t)n * sizeof(double));
        double *wi = malloc((size_t)n * sizeof(double));
        double *rr = malloc((size_t)n * sizeof(double));
        double *ri = malloc((size_t)n * sizeof(double));
        double nrm = norm2_est(A, n) / sqrt((double)n);

        sdc_stats st;
        lapack_eig_c(A, n, rr, ri);
        sdc_eigvals_c(A, n, wr, wi, 0, &st);
        double e = spec_err(wr, wi, rr, ri, n, nrm);
        if (!(e < 1e-8)) {
            printf("  n=%lld ACCURACY FAIL dlam=%.1e -- not timed\n",
                   (long long)n, e);
            free(A); free(wr); free(wi); free(rr); free(ri);
            continue;
        }

        double ts = 1e30, tl = 1e30, c0 = 1e30, c1 = 1e30;
        for (int k = 0; k < reps; k++) {                 /* contamination pre */
            double t = now_s(); lapack_eig_c(A, n, rr, ri);
            t = now_s() - t; if (t < c0) c0 = t;
        }
        sdc_stats acc; memset(&acc, 0, sizeof(acc));
        for (int k = 0; k < reps; k++) {                 /* interleaved */
            sdc_stats s1;
            double t = now_s(); sdc_eigvals_c(A, n, wr, wi, 0, &s1);
            t = now_s() - t;
            if (t < ts) { ts = t; acc = s1; }
            t = now_s(); lapack_eig_c(A, n, rr, ri);
            t = now_s() - t; if (t < tl) tl = t;
        }
        for (int k = 0; k < reps; k++) {                 /* contamination post */
            double t = now_s(); lapack_eig_c(A, n, rr, ri);
            t = now_s() - t; if (t < c1) c1 = t;
        }
        double drift = fabs(c1 - c0) / c0;
        printf("  n=%4lld  sdc_c %8.1f ms = %5.2fx dgeev   dgeev %7.1f ms"
               "   dlam %.1e   contamination %.1f%% %s\n",
               (long long)n, ts * 1e3, tl / ts, tl * 1e3, e,
               drift * 100.0, drift > 0.10 ? "CONTAMINATED" : "ok");
        printf("        phases: sign %6.1f ms (%4.1f%%)  qr %5.1f ms  "
               "gemm %5.1f ms  leaf %6.1f ms  |  %d Newton + %d NS steps, "
               "%d sign calls\n",
               acc.t_sign * 1e3, 100.0 * acc.t_sign / ts, acc.t_qr * 1e3,
               acc.t_gemm * 1e3, acc.t_leaf * 1e3, acc.n_newton, acc.n_ns,
               acc.n_sign_calls);

        /* One isolated sign call, to check the in-solve attribution against a
         * measurement that shares none of its bookkeeping (#25's lesson: a
         * decomposition whose parts contradict an isolated measurement of the
         * same kernel is measuring the harness). */
        double tsig = 1e30;
        double tr = 0.0;
        for (bi i = 0; i < n; i++) tr += A[IDX(i, i, n)];
        sdc_stats sb;
        matrix_sign_bench(A, n, tr / (double)n, &sb);
        for (int k = 0; k < reps; k++) {
            double t = now_s();
            matrix_sign_bench(A, n, tr / (double)n, &sb);
            t = now_s() - t; if (t < tsig) tsig = t;
        }
        printf("        isolated matrix_sign: %6.1f ms (%d Newton + %d NS) "
               "= %.2fx dgeev   [in-solve says %.1f ms over %d calls]\n",
               tsig * 1e3, sb.n_newton, sb.n_ns, tsig / tl, acc.t_sign * 1e3,
               acc.n_sign_calls);

        free(A); free(wr); free(wi); free(rr); free(ri);
    }

    printf("\nload at end: %.2f\n", loadavg());
    return 0;
}
