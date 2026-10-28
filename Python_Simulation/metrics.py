"""Error metrics between an approximation and the exact reference.

All functions take arrays of identical shape. Softmax metrics treat the last
axis as one probability distribution per row.
"""
import numpy as np


def abs_error(approx, ref):
    return np.abs(np.asarray(approx) - np.asarray(ref))


def mae(approx, ref):
    return float(np.mean(abs_error(approx, ref)))


def max_abs_error(approx, ref):
    return float(np.max(abs_error(approx, ref)))


def rmse(approx, ref):
    return float(np.sqrt(np.mean((np.asarray(approx) - np.asarray(ref)) ** 2)))


def relative_error(approx, ref, threshold=1e-6):
    """Mean and max relative error over entries with |ref| > threshold.

    Relative error is meaningless near zero, so those entries are excluded and
    the fraction actually evaluated is returned alongside, to keep it visible.
    """
    approx, ref = np.asarray(approx), np.asarray(ref)
    mask = np.abs(ref) > threshold
    if not mask.any():
        return {"mean": float("nan"), "max": float("nan"), "fraction_evaluated": 0.0}
    rel = np.abs(approx[mask] - ref[mask]) / np.abs(ref[mask])
    return {"mean": float(rel.mean()), "max": float(rel.max()), "fraction_evaluated": float(mask.mean())}


def sum_error(probs):
    """Max over rows of |sum(p) - 1|: how far each output is from a valid distribution."""
    return float(np.max(np.abs(np.sum(probs, axis=-1) - 1.0)))


def kl_divergence(ref, approx, eps=1e-12):
    """Mean over rows of KL(ref || approx). eps guards log(0) and is reported with results."""
    p = np.clip(ref, eps, None)
    q = np.clip(approx, eps, None)
    return float(np.mean(np.sum(p * np.log(p / q), axis=-1)))


def total_variation(ref, approx):
    """Mean over rows of 0.5 * ||ref - approx||_1, in [0, 1]."""
    return float(np.mean(0.5 * np.sum(np.abs(ref - approx), axis=-1)))


def argmax_match(ref, approx):
    """Fraction of rows whose largest element is the same in both."""
    return float(np.mean(np.argmax(ref, axis=-1) == np.argmax(approx, axis=-1)))


def softmax_report(approx, ref):
    """All Softmax metrics in one dict (one row of the results table)."""
    rel = relative_error(approx, ref)
    return {
        "mae": mae(approx, ref),
        "max_abs": max_abs_error(approx, ref),
        "rmse": rmse(approx, ref),
        "rel_mean": rel["mean"],
        "rel_max": rel["max"],
        "rel_fraction": rel["fraction_evaluated"],
        "sum_err": sum_error(approx),
        "kl": kl_divergence(ref, approx),
        "tv": total_variation(ref, approx),
        "argmax_match": argmax_match(ref, approx),
    }
