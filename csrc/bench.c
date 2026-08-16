/* Benchmark + accuracy battery for the C purification eigensolver.
 *
 * Both solvers run through the SAME binary and the SAME OpenBLAS, so the
 * comparison isolates the algorithm and its implementation, not the BLAS
 * (SSJ_LOG #13's cross-library confound).
 *
 * Every timed configuration is accuracy-checked BEFORE it is timed: a
 * routine that fails fast looks fast (SSJ_LOG #5/#7).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int64_t bi;

void purify_eigh_c(const double *A, bi n, double *w, double *V, int pairs);
void lapack_eigh_c(const double *A, bi n, double *w, double *V);

extern void scipy_dsymm_64_(const char *, const char *, const bi *, const bi *,
                            const double *, const double *, const bi *,
                            const double *, const bi *, const double *,
                            double *, const bi *);
extern void scipy_dgemm_64_(const char *, const char *, const bi *, const bi *,
                            const bi *, const double *, const double *,
                            const bi *, const double *, const bi *,
                            const double *, double *, const bi *);

#define IDX(i, j, ld) ((size_t)(i) + (size_t)(ld) * (size_t)(j))

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
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

static void goe(double *A, bi n, uint64_t seed) {
    rs = seed ? seed : 1;
    double s = 1.0 / sqrt(2.0 * (double)n);
    for (bi j = 0; j < n; j++)
        for (bi i = j; i < n; i++) {
            double v = (rnd() + rnd()) * s;
            A[IDX(i, j, n)] = v; A[IDX(j, i, n)] = v;
        }
}

/* A = Q diag(v) Q^T for a random orthogonal Q, so we can plant spectra. */
static void planted(double *A, bi n, const double *vals, uint64_t seed) {
    rs = seed ? seed : 1;
    double *M = malloc((size_t)n * n * sizeof(double));
    for (size_t k = 0; k < (size_t)n * n; k++) M[k] = rnd();
    /* Gram-Schmidt (n <= 800 here; O(n^3) but off the timed path) */
    for (bi j = 0; j < n; j++) {
        for (bi k = 0; k < j; k++) {
            double d = 0.0;
            for (bi i = 0; i < n; i++) d += M[IDX(i, k, n)] * M[IDX(i, j, n)];
            for (bi i = 0; i < n; i++) M[IDX(i, j, n)] -= d * M[IDX(i, k, n)];
        }
        double nr = 0.0;
        for (bi i = 0; i < n; i++) nr += M[IDX(i, j, n)] * M[IDX(i, j, n)];
        nr = 1.0 / sqrt(nr);
        for (bi i = 0; i < n; i++) M[IDX(i, j, n)] *= nr;
    }
    double *T = malloc((size_t)n * n * sizeof(double));
    for (bi j = 0; j < n; j++)
        for (bi i = 0; i < n; i++) T[IDX(i, j, n)] = M[IDX(i, j, n)] * vals[j];
    const double one = 1.0, zero = 0.0;
    scipy_dgemm_64_("N", "T", &n, &n, &n, &one, T, &n, M, &n, &zero, A, &n);
    for (bi j = 0; j < n; j++)
        for (bi i = 0; i < j; i++) {
            double v = 0.5 * (A[IDX(i, j, n)] + A[IDX(j, i, n)]);
            A[IDX(i, j, n)] = v; A[IDX(j, i, n)] = v;
        }
    free(M); free(T);
}

static int dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* max |lambda_i - lambda_i^ref| / ||A||, max residual / ||A||, ||V^T V - I|| */
static void metrics(const double *A, bi n, const double *w, const double *V,
                    const double *wref, double *dl, double *res, double *orth) {
    /* exact ||A||_2 = max|lambda|, taken from the dsyevd reference, so the
     * relative errors use the same convention as the Python battery */
    double nrm = fabs(wref[0]) > fabs(wref[n - 1]) ? fabs(wref[0])
                                                   : fabs(wref[n - 1]);
    if (nrm < 1e-300) nrm = 1e-300;
    double *ws = malloc((size_t)n * sizeof(double));
    memcpy(ws, w, (size_t)n * sizeof(double));
    qsort(ws, (size_t)n, sizeof(double), dcmp);
    double m = 0.0;
    for (bi i = 0; i < n; i++) { double d = fabs(ws[i] - wref[i]); if (d > m) m = d; }
    *dl = m / nrm;
    free(ws);

    double *AV = malloc((size_t)n * n * sizeof(double));
    const double one = 1.0, zero = 0.0;
    scipy_dsymm_64_("L", "L", &n, &n, &one, A, &n, V, &n, &zero, AV, &n);
    double mr = 0.0;
    for (bi j = 0; j < n; j++) {
        double s = 0.0;
        for (bi i = 0; i < n; i++) {
            double d = AV[IDX(i, j, n)] - V[IDX(i, j, n)] * w[j];
            s += d * d;
        }
        s = sqrt(s);
        if (s > mr) mr = s;
    }
    *res = mr / nrm;

    double *G = malloc((size_t)n * n * sizeof(double));
    scipy_dgemm_64_("T", "N", &n, &n, &n, &one, V, &n, V, &n, &zero, G, &n);
    double so = 0.0;
    for (bi j = 0; j < n; j++)
        for (bi i = 0; i < n; i++) {
            double d = G[IDX(i, j, n)] - (i == j ? 1.0 : 0.0);
            so += d * d;
        }
    *orth = sqrt(so);
    free(AV); free(G);
}

static double loadavg(void) {
    FILE *f = fopen("/proc/loadavg", "r");
    if (!f) return -1.0;
    double l = -1.0;
    if (fscanf(f, "%lf", &l) != 1) l = -1.0;
    fclose(f);
    return l;
}

int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 5;
    bi sizes[] = {400, 800, 1600};
    int nsizes = 3;

    printf("load at start: %.2f\n\n", loadavg());

    /* ---------------- accuracy battery (load-immune) ------------------- */
    printf("=== accuracy battery (C solver, vs dsyevd through the same binary)\n");
    printf("  %-24s %10s %10s %10s\n", "case", "dlam", "resid", "ortho");
    for (int c = 0; c < 5; c++) {
        bi n = (c == 1) ? 800 : (c == 0 ? 400 : 200);
        double *A = malloc((size_t)n * n * sizeof(double));
        double *vals = malloc((size_t)n * sizeof(double));
        const char *name;
        if (c == 0) { goe(A, n, 1); name = "GOE n=400"; }
        else if (c == 1) { goe(A, n, 1); name = "GOE n=800"; }
        else if (c == 2) {                       /* exact 5-fold ties */
            rs = 2;
            for (bi i = 0; i < n; i++) vals[i] = 0.0;
            for (bi i = 0; i < n; i += 5) {
                double v = rnd();
                for (bi k = 0; k < 5 && i + k < n; k++) vals[i + k] = v;
            }
            planted(A, n, vals, 7); name = "5-fold deg n=200";
        } else if (c == 3) {                     /* 1e-9 cluster */
            rs = 3;
            for (bi i = 0; i < n; i++) vals[i] = rnd();
            qsort(vals, (size_t)n, sizeof(double), dcmp);
            for (bi k = 0; k < 5; k++) vals[n / 2 + k] = vals[n / 2] + 1e-9 * k;
            planted(A, n, vals, 9); name = "clustered 1e-9 n=200";
        } else {                                 /* zero diagonal */
            goe(A, n, 4);
            for (bi i = 0; i < n; i++) A[IDX(i, i, n)] = 0.0;
            name = "zero diagonal n=200";
        }
        double *w = malloc((size_t)n * sizeof(double));
        double *V = malloc((size_t)n * n * sizeof(double));
        double *wref = malloc((size_t)n * sizeof(double));
        double *Vref = malloc((size_t)n * n * sizeof(double));
        lapack_eigh_c(A, n, wref, Vref);
        purify_eigh_c(A, n, w, V, 2);
        double dl, res, orth;
        metrics(A, n, w, V, wref, &dl, &res, &orth);
        /* c==3 is the 1e-9 cluster: the fp32 split route has a documented
         * intra-cluster residual floor (SSJ_LOG #19), so it carries the
         * documented bar rather than a tighter one that would "fail" a
         * behaviour the log already establishes and pins in the Python suite */
        double bar_dl = (c == 3) ? 1e-9 : 1e-11;
        double bar_res = (c == 3) ? 1e-8 : 1e-10;
        printf("  %-24s %10.1e %10.1e %10.1e%s\n", name, dl, res, orth,
               (dl < bar_dl && res < bar_res && orth < 1e-9) ? ""
                                                            : "   <-- FAIL");
        free(A); free(vals); free(w); free(V); free(wref); free(Vref);
    }

    /* ------------------------- the wall race --------------------------- */
    printf("\n=== wall (interleaved min-of-%d, accuracy asserted first)\n", reps);
    for (int si = 0; si < nsizes; si++) {
        bi n = sizes[si];
        double *A = malloc((size_t)n * n * sizeof(double));
        goe(A, n, 1);
        double *w = malloc((size_t)n * sizeof(double));
        double *V = malloc((size_t)n * n * sizeof(double));
        double *wref = malloc((size_t)n * sizeof(double));
        double *Vref = malloc((size_t)n * n * sizeof(double));

        lapack_eigh_c(A, n, wref, Vref);
        purify_eigh_c(A, n, w, V, 2);
        double dl, res, orth;
        metrics(A, n, w, V, wref, &dl, &res, &orth);
        if (!(dl < 1e-11 && res < 1e-10 && orth < 1e-9)) {
            printf("  n=%lld ACCURACY FAIL dlam=%.1e res=%.1e orth=%.1e\n",
                   (long long)n, dl, res, orth);
            free(A); free(w); free(V); free(wref); free(Vref);
            continue;
        }

        double tp = 1e30, tl = 1e30, c0 = 1e30, c1 = 1e30;
        purify_eigh_c(A, n, w, V, 2); lapack_eigh_c(A, n, wref, Vref);
        for (int k = 0; k < reps; k++) {                  /* contamination pre */
            double t = now(); lapack_eigh_c(A, n, wref, Vref);
            t = now() - t; if (t < c0) c0 = t;
        }
        for (int k = 0; k < reps; k++) {                  /* interleaved */
            double t = now(); purify_eigh_c(A, n, w, V, 2);
            t = now() - t; if (t < tp) tp = t;
            t = now(); lapack_eigh_c(A, n, wref, Vref);
            t = now() - t; if (t < tl) tl = t;
        }
        for (int k = 0; k < reps; k++) {                  /* contamination post */
            double t = now(); lapack_eigh_c(A, n, wref, Vref);
            t = now() - t; if (t < c1) c1 = t;
        }
        double drift = fabs(c1 - c0) / c0;
        printf("  n=%4lld  purify_eigh_c %8.1f ms = %5.2fx   dsyevd %7.1f ms"
               "   dlam %.1e res %.1e   contamination %.1f%% %s\n",
               (long long)n, tp * 1e3, tp / tl, tl * 1e3, dl, res,
               drift * 100.0, drift > 0.10 ? "CONTAMINATED" : "ok");
        free(A); free(w); free(V); free(wref); free(Vref);
    }
    printf("\nload at end: %.2f\n", loadavg());
    return 0;
}
