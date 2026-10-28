"""ctypes bridge to the C approximation library (C_Simulation/build).

Python never re-implements the approximation: it passes NumPy arrays to C in
one batch call and gets the approximate results back.
"""
import ctypes
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_LIB_NAME = "dlha_approx.dll" if sys.platform == "win32" else "libdlha_approx.so"
LIB_PATH = _ROOT / "C_Simulation" / "build" / _LIB_NAME

_ERRORS = {
    -1: "invalid argument (h must divide [x_min, x_max] into an integer number of segments)",
    -2: "numerical error (non-positive exponential sum)",
}

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not LIB_PATH.exists():
        raise FileNotFoundError(f"{LIB_PATH} not found. Build it first: cd C_Simulation && mingw32-make")

    lib = ctypes.CDLL(str(LIB_PATH))
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")

    lib.softmax_approx_batch.argtypes = [f64, f64, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_int]
    lib.softmax_approx_batch.restype = ctypes.c_int

    lib.pla_exp_batch.argtypes = [f64, f64, ctypes.c_int,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.pla_exp_batch.restype = ctypes.c_int

    _lib = lib
    return lib


def _check(rc):
    if rc != 0:
        raise ValueError(f"C library returned {rc}: {_ERRORS.get(rc, 'unknown error')}")


def softmax_approx(x, h=1.0, x_min=-16.0, x_max=16.0, subtract_max=False):
    """Approximate Softmax along the last axis. x: 1-D vector or 2-D (rows, n) array."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError("x must be 1-D or 2-D")
    x2 = x.reshape(1, -1) if x.ndim == 1 else x
    out = np.empty_like(x2)
    rows, cols = x2.shape
    _check(_load().softmax_approx_batch(x2, out, rows, cols, h, x_min, x_max, int(subtract_max)))
    return out.reshape(x.shape)


def pla_exp(x, h=1.0, x_min=-16.0, x_max=16.0):
    """Element-wise PLA approximation of e^x (any shape)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    out = np.empty_like(flat)
    _check(_load().pla_exp_batch(flat, out, flat.size, h, x_min, x_max))
    return out.reshape(x.shape)


if __name__ == "__main__":
    v = np.array([-3.0, -0.5, 0.0, 1.2, 2.7])
    for h in (0.5, 1.0, 2.0, 4.0):
        y = softmax_approx(v, h=h)
        print(f"h={h:<4} softmax_approx={np.round(y, 4)}  sum={y.sum():.12f}")
