#ifndef SOFTMAX_APPROX_H
#define SOFTMAX_APPROX_H

/*
 * Approximate Softmax -- floating-point stage.
 *
 * Per row of length n:
 *   1. (optional, [OURS]) z_j = x_j - max(x)        -- not part of the paper
 *   2. e_j = PLA(z_j)                               -- [PAPER] Sec. 4.1, SM_EXP
 *   3. S   = sum_j e_j                              -- [PAPER] adder tree + accumulator
 *   4. r   = 1 / S                                  -- see note below
 *   5. y_j = e_j * r                                -- [PAPER] division as a * (1/b), SM_DIV
 *
 * Note on step 4: [PAPER] Sec. 5.3 obtains 1/S from a reciprocal LUT
 * (log-uniform over [1e-6, 1]). In this floating-point stage we use an exact
 * reciprocal so that the error measured here is the PLA error alone.
 * The reciprocal LUT will be added later as a separate, switchable stage.
 */

#include "pla_exp.h"

#if defined(_WIN32) && defined(DLHA_BUILD_DLL)
#define DLHA_API __declspec(dllexport)
#else
#define DLHA_API
#endif

/* One row, using a prebuilt table. in and out may point to the same buffer. */
int softmax_approx_row(const pla_table_t *t, const double *in, double *out,
                       int n, int subtract_max);

/*
 * Batch entry point for Python (ctypes).
 * in/out: row-major arrays of rows * cols doubles.
 * Returns DLHA_OK, DLHA_ERR_ARG (bad sizes or h), or DLHA_ERR_NUMERIC.
 */
DLHA_API int softmax_approx_batch(const double *in, double *out, int rows, int cols,
                                  double h, double x_min, double x_max,
                                  int subtract_max);

/* Element-wise PLA of e^x, exposed so Python can plot the exp approximation itself. */
DLHA_API int pla_exp_batch(const double *in, double *out, int n,
                           double h, double x_min, double x_max);

#endif /* SOFTMAX_APPROX_H */
