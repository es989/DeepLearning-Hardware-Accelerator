# DeepLearning-Hardware-Accelerator

Final-year project: hardware accelerator for Softmax and LayerNorm on FPGA.

Reference paper: R. Kim et al., "Hardware Accelerator for Approximation-Based Softmax and
Layer Normalization in Transformers", *Electronics* 2025, 14(12), 2337.
https://doi.org/10.3390/electronics14122337

## Division of work

- **C_Simulation/**: only the hardware-oriented approximations (no exact reference in C).
- **Python_Simulation/**: the exact NumPy reference, error metrics and plots. It calls the C
  library through `c_bridge.py` (ctypes, one call per batch).

## Current stage: Softmax approximation, floating point

| Item | Source |
| --- | --- |
| e^x by PLA on [-16, 16], uniform segments of width h | [PAPER] Sec. 4.1 |
| Chord coefficients (line through e^{x_i} and e^{x_{i+1}}) | **Our reconstruction**: the paper does not publish its coefficients; chord reproduces its 8.62% at h = 1 |
| Clamping of inputs outside [-16, 16] | Our decision (not specified in the paper) |
| Division as multiply by 1/S | [PAPER] Sec. 5.3 |
| 1/S exact (no reciprocal LUT yet) | Our decision for this stage, to isolate the PLA error |
| Optional max-subtraction (`subtract_max`, default off) | Our addition, not in the paper |

`h` must divide 32 (e.g. 0.25, 0.5, 1, 2, 4, 8).

## Build and run (Windows)

Requires GCC (MSYS2 UCRT64: `pacman -S mingw-w64-ucrt-x86_64-gcc make`). From Git Bash or the
MSYS2 terminal:

```sh
cd C_Simulation
mingw32-make          # builds build/dlha_approx.dll and the C sanity test
mingw32-make test     # runs the C sanity test
cd ..
python Python_Simulation/c_bridge.py        # smoke test through ctypes
python Python_Simulation/test_softmax_c.py   # C vs. Python model, edge cases, paper's 8.62% at h=1
python Python_Simulation/compare_softmax.py  # h sweep vs. exact Softmax -> Results/softmax_float/
```

| Python file | Role |
| --- | --- |
| `softmax_reference.py` | Exact Softmax (golden reference); NumPy chord-PLA model used only to verify the C code |
| `metrics.py` | MAE, max abs, RMSE, relative error (entries above a threshold), sum error, KL, TV, argmax match |
| `test_softmax_c.py` | Correctness tests of the C library |
| `compare_softmax.py` | Sweep over h and max-subtraction on 6 input distributions (seed 0): CSV + plots |

From Python:

```python
from c_bridge import softmax_approx, pla_exp
y = softmax_approx(x, h=1.0)       # x: shape (n,) or (rows, n), float64
e = pla_exp(x, h=4.0)
```
