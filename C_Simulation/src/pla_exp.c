#include "pla_exp.h"

#include <math.h>
#include <stddef.h>

int pla_table_build(pla_table_t *t, double x_min, double x_max, double h)
{
    if (t == NULL || !(h > 0.0) || !(x_max > x_min))
        return DLHA_ERR_ARG;

    double ratio = (x_max - x_min) / h;
    int n_seg = (int)(ratio + 0.5);
    if (n_seg < 1 || n_seg > PLA_MAX_SEGMENTS || fabs(ratio - n_seg) > 1e-9 * ratio)
        return DLHA_ERR_ARG;

    t->x_min = x_min;
    t->x_max = x_max;
    t->h = h;
    t->n_seg = n_seg;

    /*
     * Offline coefficient generation: this is the only place exp() is used.
     * It plays the role of precomputing the contents of exponential_mem;
     * the approximation path (pla_exp_eval) never calls exp().
     */
    for (int i = 0; i < n_seg; i++) {
        double x0 = x_min + i * h;
        double x1 = x0 + h;
        double y0 = exp(x0);
        double y1 = exp(x1);
        t->w[i] = (y1 - y0) / h;
        t->b[i] = y0 - t->w[i] * x0;
    }
    return DLHA_OK;
}

double pla_exp_eval(const pla_table_t *t, double x)
{
    if (x < t->x_min)
        x = t->x_min;
    if (x > t->x_max)
        x = t->x_max;

    /* x >= x_min here, so truncation equals floor. */
    int i = (int)((x - t->x_min) / t->h);
    if (i >= t->n_seg)
        i = t->n_seg - 1;   /* x == x_max belongs to the last segment */

    return t->w[i] * x + t->b[i];
}
