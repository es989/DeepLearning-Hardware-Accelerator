"""Compare the C Softmax approximation (floating point) against the exact reference.

Sweeps the PLA interval h and max-subtraction over several input distributions,
writes a metrics table and plots to Results/softmax_float/.

Run: python Python_Simulation/compare_softmax.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from c_bridge import pla_exp, softmax_approx
from metrics import softmax_report
from softmax_reference import softmax_exact

SEED = 0
H_VALUES = (0.5, 1.0, 2.0, 4.0)
OUT_DIR = Path(__file__).resolve().parent.parent / "Results" / "softmax_float"


def make_input_sets(rng):
    """Named input matrices (rows, n). Each row is one Softmax vector."""
    return {
        "normal_s1_n50": rng.normal(0.0, 1.0, (2000, 50)),
        "normal_s3_n50": rng.normal(0.0, 3.0, (2000, 50)),
        "uniform_pm8_n50": rng.uniform(-8.0, 8.0, (2000, 50)),
        "negative_n50": rng.uniform(-12.0, -2.0, (2000, 50)),
        "wide_s10_n50": rng.normal(0.0, 10.0, (2000, 50)),   # many inputs outside [-16, 16]
        "normal_s1_n512": rng.normal(0.0, 1.0, (200, 512)),
    }


def run_sweep(input_sets):
    rows = []
    for set_name, x in input_sets.items():
        ref = softmax_exact(x)
        for h in H_VALUES:
            for subtract_max in (False, True):
                approx = softmax_approx(x, h=h, subtract_max=subtract_max)
                rows.append({"input": set_name, "n": x.shape[1], "h": h,
                             "subtract_max": int(subtract_max), **softmax_report(approx, ref)})
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print(f"{'input':<18}{'h':>5}{'maxsub':>8}{'max_abs':>10}{'tv':>9}{'kl':>10}{'sum_err':>10}{'argmax':>8}")
    for r in rows:
        print(f"{r['input']:<18}{r['h']:>5}{r['subtract_max']:>8}{r['max_abs']:>10.4f}"
              f"{r['tv']:>9.4f}{r['kl']:>10.5f}{r['sum_err']:>10.1e}{100 * r['argmax_match']:>7.1f}%")


def plot_pla_exp(path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.linspace(-16.0, 16.0, 32001)
    ax1.semilogy(x, np.exp(x), "k", lw=2, label="exact $e^x$")
    for h in H_VALUES:
        ax1.semilogy(x, pla_exp(x, h=h), lw=1, label=f"PLA h={h}")
    ax1.set(title="Chord PLA of $e^x$ on [-16, 16]", xlabel="x", ylabel="value (log scale)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    xz = np.linspace(-4.0, 4.0, 8001)
    for h in H_VALUES:
        rel = (pla_exp(xz, h=h) - np.exp(xz)) / np.exp(xz)
        ax2.plot(xz, 100 * rel, lw=1, label=f"h={h}")
    ax2.set(title="Relative error of PLA (periodic in each segment)", xlabel="x", ylabel="relative error [%]")
    ax2.set_yscale("symlog", linthresh=1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_vs_h(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    inputs = list(dict.fromkeys(r["input"] for r in rows))
    colors = plt.cm.tab10(np.arange(len(inputs)))
    for metric, ax in zip(("tv", "max_abs"), axes):
        for name, color in zip(inputs, colors):
            for subtract_max, style in ((0, "-o"), (1, "--s")):
                sel = [r for r in rows if r["input"] == name and r["subtract_max"] == subtract_max]
                ax.plot([r["h"] for r in sel], [r[metric] for r in sel], style, ms=4, color=color,
                        label=f"{name}{' +maxsub' if subtract_max else ''}")
        ax.set(xscale="log", yscale="log", xlabel="PLA interval h", ylabel=metric,
               title=f"Softmax {metric} vs h (solid: no max-sub, dashed: max-sub)")
        ax.set_xticks(H_VALUES, [str(h) for h in H_VALUES])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example_row(x_row, path):
    order = np.argsort(x_row)
    ref = softmax_exact(x_row)[order]
    idx = np.arange(x_row.size)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(idx, ref, color="lightgray", label="exact")
    for h, marker in ((1.0, "o"), (4.0, "x")):
        ax.plot(idx, softmax_approx(x_row, h=h)[order], marker, ms=5, label=f"PLA h={h}")
    ax.set(title="One Softmax row (elements sorted by input)", xlabel="element (sorted)", ylabel="probability")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    input_sets = make_input_sets(rng)

    rows = run_sweep(input_sets)
    print_table(rows)
    write_csv(rows, OUT_DIR / "metrics.csv")

    plot_pla_exp(OUT_DIR / "pla_exp.png")
    plot_metrics_vs_h(rows, OUT_DIR / "metrics_vs_h.png")
    plot_example_row(input_sets["normal_s1_n50"][0], OUT_DIR / "example_row.png")
    print(f"\nSeed {SEED}. Results written to {OUT_DIR}")


if __name__ == "__main__":
    main()
