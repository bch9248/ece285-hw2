"""Plot training loss curves from runs/<name>/losses.csv.

    python plot_losses.py --run runs/dcgan64
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(x, k):
    if k <= 1 or len(x) < k:
        return np.asarray(x)
    return np.convolve(x, np.ones(k) / k, mode="valid")


def plot(csv_path, out_prefix, window=100):
    cols = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    step = np.array(cols["step"])
    w = min(window, max(1, len(step) // 10))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for key, label in [("loss_D", "Discriminator"), ("loss_G", "Generator")]:
        axes[0].plot(step, cols[key], alpha=0.15)
        axes[0].plot(step[w - 1:], smooth(cols[key], w), label=label, color=axes[0].lines[-1].get_color())
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss"); axes[0].set_title("Training losses")
    axes[0].legend()
    for key, label in [("D_x", "D(x)  real"), ("D_G_z", "D(G(z))  fake")]:
        axes[1].plot(step[w - 1:], smooth(cols[key], w), label=label)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("mean D output"); axes[1].set_title("Discriminator outputs")
    axes[1].legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved {out_prefix}.png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory containing losses.csv")
    p.add_argument("--window", type=int, default=100, help="moving-average window")
    a = p.parse_args()
    plot(os.path.join(a.run, "losses.csv"), os.path.join(a.run, "loss_curves"), a.window)
