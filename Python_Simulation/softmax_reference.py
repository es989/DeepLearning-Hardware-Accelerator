"""Exact Softmax in float64: the golden reference for the C approximation."""
import numpy as np


def softmax_exact(x, axis=-1):
    """Numerically stable Softmax: exp(x - max) / sum(exp(x - max)).

    Subtracting the max does not change the mathematical result; it only keeps
    np.exp from overflowing, so this is the exact function for any input range.
    """
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def pla_exp_chord(x, h=1.0, x_min=-16.0, x_max=16.0):
    """Independent NumPy model of the chord PLA, used only to verify the C code.

    It mirrors the documented algorithm (clamp, segment index, w*x + b), so the
    C output must match it to floating-point rounding. It is not a second
    approximation implementation for experiments; those go through C.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), x_min, x_max)
    n_seg = int(round((x_max - x_min) / h))
    i = np.minimum(((x - x_min) / h).astype(int), n_seg - 1)
    x0 = x_min + i * h
    w = (np.exp(x0 + h) - np.exp(x0)) / h
    b = np.exp(x0) - w * x0
    return w * x + b
