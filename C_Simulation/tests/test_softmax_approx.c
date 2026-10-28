/*
 * Structural sanity tests for the C approximation.
 * Numerical accuracy against the exact Softmax is tested in Python (golden reference);
 * exp() appears here only to check that the table hits e^x exactly at segment nodes.
 */
#include "softmax_approx.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond, msg)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            printf("FAIL: %s (line %d)\n", msg, __LINE__);  \
            failures++;                                     \
        }                                                   \
    } while (0)

static void test_table_rejects_bad_h(void)
{
    pla_table_t t;
    CHECK(pla_table_build(&t, -16.0, 16.0, 3.0) == DLHA_ERR_ARG, "h=3 does not divide [-16,16]");
    CHECK(pla_table_build(&t, -16.0, 16.0, 0.0) == DLHA_ERR_ARG, "h=0 rejected");
    CHECK(pla_table_build(&t, -16.0, 16.0, 0.5) == DLHA_OK, "h=0.5 accepted");
    CHECK(t.n_seg == 64, "h=0.5 gives 64 segments");
}

static void test_pla_exact_at_nodes(void)
{
    pla_table_t t;
    pla_table_build(&t, -16.0, 16.0, 1.0);
    for (int k = -16; k <= 16; k += 4) {
        double x = (double)k;
        double rel = fabs(pla_exp_eval(&t, x) - exp(x)) / exp(x);
        CHECK(rel < 1e-12, "chord PLA equals e^x at segment nodes");
    }
}

static void test_pla_clamps_out_of_range(void)
{
    pla_table_t t;
    pla_table_build(&t, -16.0, 16.0, 1.0);
    CHECK(pla_exp_eval(&t, -100.0) == pla_exp_eval(&t, -16.0), "clamp below x_min");
    CHECK(pla_exp_eval(&t, 100.0) == pla_exp_eval(&t, 16.0), "clamp above x_max");
}

static void test_softmax_properties(void)
{
    const double x[5] = {-3.0, -0.5, 0.0, 1.2, 2.7};
    double y[5];
    double h_values[4] = {0.5, 1.0, 2.0, 4.0};

    for (int k = 0; k < 4; k++) {
        for (int sm = 0; sm <= 1; sm++) {
            int rc = softmax_approx_batch(x, y, 1, 5, h_values[k], -16.0, 16.0, sm);
            CHECK(rc == DLHA_OK, "batch returns OK");
            double sum = 0.0;
            for (int j = 0; j < 5; j++)
                sum += y[j];
            CHECK(fabs(sum - 1.0) < 1e-12, "outputs sum to 1");
            for (int j = 1; j < 5; j++)
                CHECK(y[j] > y[j - 1], "ascending input gives ascending output");
        }
    }
}

static void test_softmax_edge_cases(void)
{
    double one = 3.0, y1;
    CHECK(softmax_approx_batch(&one, &y1, 1, 1, 1.0, -16.0, 16.0, 0) == DLHA_OK, "n=1 OK");
    CHECK(fabs(y1 - 1.0) < 1e-12, "n=1 gives 1");

    const double same[4] = {2.0, 2.0, 2.0, 2.0};
    double y[4];
    softmax_approx_batch(same, y, 1, 4, 4.0, -16.0, 16.0, 0);
    for (int j = 0; j < 4; j++)
        CHECK(fabs(y[j] - 0.25) < 1e-12, "identical inputs give 1/n");

    CHECK(softmax_approx_batch(same, y, 1, 0, 1.0, -16.0, 16.0, 0) == DLHA_ERR_ARG, "cols=0 rejected");
}

int main(void)
{
    test_table_rejects_bad_h();
    test_pla_exact_at_nodes();
    test_pla_clamps_out_of_range();
    test_softmax_properties();
    test_softmax_edge_cases();

    if (failures == 0)
        printf("All tests passed.\n");
    return failures == 0 ? 0 : 1;
}
