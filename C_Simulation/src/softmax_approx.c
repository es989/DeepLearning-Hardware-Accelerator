#include "softmax_approx.h"

#include <stddef.h>

int softmax_approx_row(const pla_table_t *t, const double *in, double *out,
                       int n, int subtract_max)
{
    if (t == NULL || in == NULL || out == NULL || n < 1)
        return DLHA_ERR_ARG;

    double shift = 0.0;
    if (subtract_max) {
        shift = in[0];
        for (int j = 1; j < n; j++)
            if (in[j] > shift)
                shift = in[j];
    }

    /* SM_EXP: PLA per element, accumulated into S. */
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
        out[j] = pla_exp_eval(t, in[j] - shift);
        sum += out[j];
    }

    /* Chord segments of e^x are positive, so S > 0 unless something is broken. */
    if (!(sum > 0.0))
        return DLHA_ERR_NUMERIC;

    /* SM_DIV: multiply by the reciprocal (exact in this stage). */
    double recip = 1.0 / sum;
    for (int j = 0; j < n; j++)
        out[j] *= recip;

    return DLHA_OK;
}

int softmax_approx_batch(const double *in, double *out, int rows, int cols,
                         double h, double x_min, double x_max,
                         int subtract_max)
{
    if (in == NULL || out == NULL || rows < 0 || cols < 1)
        return DLHA_ERR_ARG;

    pla_table_t table;
    int rc = pla_table_build(&table, x_min, x_max, h);
    if (rc != DLHA_OK)
        return rc;

    for (int r = 0; r < rows; r++) {
        size_t offset = (size_t)r * (size_t)cols;
        rc = softmax_approx_row(&table, in + offset, out + offset, cols, subtract_max);
        if (rc != DLHA_OK)
            return rc;
    }
    return DLHA_OK;
}

int pla_exp_batch(const double *in, double *out, int n,
                  double h, double x_min, double x_max)
{
    if (in == NULL || out == NULL || n < 0)
        return DLHA_ERR_ARG;

    pla_table_t table;
    int rc = pla_table_build(&table, x_min, x_max, h);
    if (rc != DLHA_OK)
        return rc;

    for (int j = 0; j < n; j++)
        out[j] = pla_exp_eval(&table, in[j]);
    return DLHA_OK;
}
