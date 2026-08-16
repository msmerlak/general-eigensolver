/* Public interface for the C spectral divide-and-conquer eigensolver. */
#ifndef SDC_EIG_H
#define SDC_EIG_H

#include <stdint.h>

/* Wall-clock and iteration counters, so the phase split is attributed rather
 * than inferred. Accumulated across a whole solve (SSJ_LOG #25: a per-call
 * single-shot decomposition measured the harness, not the algorithm). */
typedef struct {
    double t_sign;      /* total time inside matrix_sign                    */
    double t_qr;        /* pivoted QR + dorgqr for the split basis          */
    double t_gemm;      /* the two gemms forming Q^T A Q                    */
    double t_leaf;      /* dgeev on the leaves                              */
    int n_newton;       /* scaled-Newton steps taken (each needs an inverse)*/
    int n_ns;           /* Newton-Schulz steps taken (2 gemms, no inverse)  */
    int n_sign_calls;   /* sign evaluations, including failed shift retries */
    int n_fallbacks;    /* blocks that found no usable split                */
} sdc_stats;

/* Eigenvalues of a general real n x n matrix (column-major) by spectral
 * divide and conquer. wr/wi receive the real and imaginary parts, unsorted.
 * min_block <= 0 selects the measured default max(2, n/2) -- one split, both
 * halves to dgeev (SSJ_LOG #25: recursing to 2x2 is 4x slower, no more
 * accurate). `st` may be NULL. */
void sdc_eigvals_c(const double *A, int64_t n, double *wr, double *wi,
                   int64_t min_block, sdc_stats *st);

/* dgeev on the same BLAS, for the reference side of the race. */
void lapack_eig_c(const double *A, int64_t n, double *wr, double *wi);

/* One isolated sign evaluation, for attribution. Returns iterations, or -1. */
int matrix_sign_bench(const double *A, int64_t n, double shift, sdc_stats *st);

#endif /* SDC_EIG_H */
