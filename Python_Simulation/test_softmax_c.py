"""Correctness tests of the C Softmax approximation (run: python test_softmax_c.py, or pytest).

These check that C implements the documented algorithm and handles edge cases.
How good the approximation is, is measured separately in compare_softmax.py.
"""
import numpy as np

from c_bridge import pla_exp, softmax_approx
from softmax_reference import pla_exp_chord, softmax_exact

H_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def test_c_pla_matches_python_model():
    x = np.linspace(-20.0, 20.0, 40001)          # includes out-of-range inputs
    for h in H_VALUES:
        c, py = pla_exp(x, h=h), pla_exp_chord(x, h=h)
        assert np.max(np.abs(c - py) / py) < 1e-12, f"h={h}"


def test_c_softmax_matches_python_model():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 4.0, size=(500, 37))
    for h in H_VALUES:
        for subtract_max in (False, True):
            z = x - x.max(axis=1, keepdims=True) if subtract_max else x
            e = pla_exp_chord(z, h=h)
            expected = e / e.sum(axis=1, keepdims=True)
            got = softmax_approx(x, h=h, subtract_max=subtract_max)
            assert np.max(np.abs(got - expected)) < 1e-14, f"h={h} subtract_max={subtract_max}"


def test_paper_h1_error_reproduced():
    # [PAPER] Sec. 4.1: 8.62% mean relative error at h = 1 on [-16, 16].
    x = np.linspace(-16.0, 16.0, 32001)
    rel = np.abs(pla_exp(x, h=1.0) - np.exp(x)) / np.exp(x)
    assert abs(100 * rel.mean() - 8.62) < 0.01


def test_edge_cases():
    assert np.allclose(softmax_approx(np.array([5.0]), h=4.0), [1.0])
    assert np.allclose(softmax_approx(np.full(8, -3.0), h=2.0), np.full(8, 1 / 8))
    big = np.array([100.0, 0.0, -100.0])          # beyond [-16, 16]: clamped, no overflow
    y = softmax_approx(big, h=1.0)
    assert np.all(np.isfinite(y)) and abs(y.sum() - 1.0) < 1e-12


def test_batch_equals_row_by_row():
    x = np.random.default_rng(2).normal(size=(20, 16))
    whole = softmax_approx(x, h=1.0)
    rows = np.stack([softmax_approx(r, h=1.0) for r in x])
    assert np.array_equal(whole, rows)


def test_invalid_h_rejected():
    try:
        softmax_approx(np.zeros(4), h=3.0)
    except ValueError:
        return
    raise AssertionError("h=3 should be rejected")


def test_exact_reference_sanity():
    p = softmax_exact(np.array([[1000.0, 1000.0], [0.0, np.log(3.0)]]))
    assert np.allclose(p, [[0.5, 0.5], [0.25, 0.75]])


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)} tests passed.")
