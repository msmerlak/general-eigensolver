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
    /* Why shift attempts were rejected, and what they cost. A failed attempt
     * runs a FULL sign iteration before the guard fires, so these are the
     * expensive counters, not diagnostics. */
    int n_fail_rank;      /* trace(P) degenerate: everything on one side     */
    int n_fail_singular;  /* singular iterate / LU failure                   */
    int n_fail_resid;     /* ||A21||/||A|| over the 1e-6 backward-error bar  */
    int n_fail_maxiter;   /* sign hit max_iter without reaching tol          */
    int iters_wasted;     /* sign iterations spent inside REJECTED attempts  */
    double last_dev;      /* ||X^2-I||_F/sqrt(n) at the last sign exit       */
    int last_sign_iters;  /* iterations of the last SUCCESSFUL sign call     */
} sdc_stats;

/* Eigenvalues of a general real n x n matrix (column-major) by spectral
 * divide and conquer. wr/wi receive the real and imaginary parts, unsorted.
 * min_block <= 0 selects the measured default 3n/5 -- one split, both halves
 * to dgeev. Two measured constraints, not one: recursing to 2x2 is 4x slower
 * and no more accurate (SSJ_LOG #25), and a leaf of exactly n/2 costs a
 * SECOND sign iteration because the centred split returns r near but not on
 * n/2 (SSJ_LOG #28, worth 1.10x-1.16x). `st` may be NULL. */
void sdc_eigvals_c(const double *A, int64_t n, double *wr, double *wi,
                   int64_t min_block, sdc_stats *st);

/* dgeev on the same BLAS, for the reference side of the race. */
void lapack_eig_c(const double *A, int64_t n, double *wr, double *wi);

/* One isolated sign evaluation, for attribution. Returns iterations, or -1. */
int matrix_sign_bench(const double *A, int64_t n, double shift, sdc_stats *st);

/* Threshold on ||X^2 - I||_F/sqrt(n) below which the sign iteration hands off
 * from scaled Newton to Newton-Schulz. Exposed so it can be swept rather than
 * asserted. Default 0.9, and NOT a lever: the iteration converges
 * quadratically and crosses the whole disputed band in 1-2 of 16 steps, so
 * there is nothing to reassign wherever it sits (SSJ_LOG #29). The 0.6 -> 0.9
 * move is worth 3-5%, inside this box's contamination band. */
void sdc_set_ns_switch(double v);
double sdc_get_ns_switch(void);

#endif /* SDC_EIG_H */
