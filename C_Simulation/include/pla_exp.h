#ifndef PLA_EXP_H
#define PLA_EXP_H

/*
 * Piecewise Linear Approximation (PLA) of e^x -- floating-point stage.
 *
 * [PAPER] Kim et al., Electronics 2025, 14(12), 2337, Sec. 4.1:
 *         e^x ~= w_i * x + b_i  for x in [x_i, x_{i+1}],
 *         input domain [-16, 16] split into uniform intervals of width h,
 *         coefficients precomputed and stored in on-chip memory
 *         ("exponential_mem"); one multiply + one add per element.
 *
 * [OURS]  The paper does not publish the coefficients. We use "chord"
 *         coefficients: segment i passes exactly through e^{x_i} and
 *         e^{x_{i+1}}. For h = 1 this reproduces the paper's reported 8.62%
 *         mean relative error, but it is a reconstruction, not verified
 *         paper coefficients.
 * [OURS]  Inputs outside [x_min, x_max] are clamped (saturated). The paper
 *         does not specify out-of-range behaviour.
 */

#define PLA_MAX_SEGMENTS 1024

#define DLHA_OK          0
#define DLHA_ERR_ARG    (-1)
#define DLHA_ERR_NUMERIC (-2)

typedef struct {
    double x_min;
    double x_max;
    double h;
    int    n_seg;
    double w[PLA_MAX_SEGMENTS];   /* slope per segment     */
    double b[PLA_MAX_SEGMENTS];   /* intercept per segment */
} pla_table_t;

/*
 * Builds the coefficient table (the software model of exponential_mem).
 * h must divide (x_max - x_min) into an integer number of segments.
 * Returns DLHA_OK or DLHA_ERR_ARG.
 */
int pla_table_build(pla_table_t *t, double x_min, double x_max, double h);

/* Evaluates the PLA: segment lookup, then one multiply and one add. */
double pla_exp_eval(const pla_table_t *t, double x);

#endif /* PLA_EXP_H */
