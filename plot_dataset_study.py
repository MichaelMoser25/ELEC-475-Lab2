"""
ELEC 475: Lab 2 - Step 4 Dual-Dataset Benchmark (CIFAR-10 vs. Tiny ImageNet-200)

Compares the Tiny ImageNet run with the Step 3 augmented CIFAR-10 run. Both use AlexNet-64 + BatchNorm,
the same augmentation, optimizer and seed; the CIFAR-10 run is cut to the same number of epochs
(no learning-rate schedule, so its first N epochs are exactly an N-epoch run).

Produces:
  - plots/dataset_study.png          : one comparison figure for the report
  - plots/dataset_study_summary.csv  : the numbers behind it (also printed as a Markdown table)

Usage:
    python plot_dataset_study.py
    python plot_dataset_study.py --tiny logs/alexnet_tiny_imagenet.json --cifar logs/alexnet_with_aug.json
"""
import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_norm_study import AXIS, INK, INK_2, MUTED, SURFACE, direct_labels, style_axis

PER_EPOCH = ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "epoch_time_s", "train_acc_clean"]
INFO = {  # classes, training images per class, native resolution
    "cifar10": (10, 5000, "32×32 (upsampled to 64×64)"),
    "tiny_imagenet": (200, 500, "64×64"),
}


def load(path, name, color, n_epochs=None):
    with open(path) as f:
        h = json.load(f)
    if n_epochs:
        for k in PER_EPOCH:
            if h.get(k):
                h[k] = h[k][:n_epochs]
    classes, per_class, res = INFO[h["dataset"]]
    best = int(np.argmax(h["val_acc"]))
    min_vl = int(np.argmin(h["val_loss"]))
    clean = h.get("train_acc_clean") or None
    s = {
        "classes": classes, "per_class": per_class, "res": res, "chance": 100.0 / classes,
        "epochs": len(h["val_acc"]),
        "best_val_acc": h["val_acc"][best], "best_epoch": best + 1, "f1_at_best": h["val_f1"][best],
        "val_acc": h["val_acc"][-1], "val_f1": h["val_f1"][-1],
        "train_clean": clean[-1] if clean else None,
        "gap_clean": clean[-1] - h["val_acc"][-1] if clean else None,
        "gap_logged": h["train_acc"][-1] - h["val_acc"][-1],
        "init_loss": math.log(classes),
        "train_loss": h["train_loss"][-1], "val_loss": h["val_loss"][-1],
        "min_val_loss": h["val_loss"][min_vl], "min_val_loss_epoch": min_vl + 1,
        "mean_epoch_s": float(np.mean(h["epoch_time_s"])),
    }
    return {"name": name, "color": color, "h": h, "s": s}


def epoch_ticks(ax, n, right_pad):
    ax.set_xlabel("Epoch", fontsize=9, color=INK_2)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlim(0.5, n + right_pad)


def loss_panel(ax, r, n_ep):
    h, s, c = r["h"], r["s"], r["color"]
    e = list(range(1, len(h["val_loss"]) + 1))
    ax.plot(e, h["train_loss"], color=c, linewidth=1.4, linestyle=":", zorder=3)
    ax.plot(e, h["val_loss"], color=c, linewidth=1.8, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
    ax.plot(s["min_val_loss_epoch"], s["min_val_loss"], marker="o", markersize=11, markerfacecolor="none",
            markeredgecolor=c, markeredgewidth=1.6, zorder=5)
    style_axis(ax, f"{r['name']}: loss", "Cross-entropy")
    epoch_ticks(ax, n_ep, right_pad=2.6)
    lo, hi = ax.get_ylim()
    direct_labels(ax, [(e[-1], h["val_loss"][-1], "Validation"), (e[-1], h["train_loss"][-1], "Train")],
                  min_gap=(hi - lo) * 0.06)
    ax.annotate(f"○ = lowest val loss (epoch {s['min_val_loss_epoch']})", xy=(0.97, 0.96), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color=MUTED)


def main():
    p = argparse.ArgumentParser(description="Plot the CIFAR-10 vs Tiny ImageNet comparison")
    p.add_argument("--tiny", default="logs/alexnet_tiny_imagenet.json")
    p.add_argument("--cifar", default="logs/alexnet_with_aug.json", help="Step 3 augmented CIFAR-10 run")
    p.add_argument("--out", default="plots/dataset_study.png")
    args = p.parse_args()

    tiny = load(args.tiny, "Tiny ImageNet-200", "#6250d6")
    n_ep = tiny["s"]["epochs"]
    cifar = load(args.cifar, "CIFAR-10", "#eb6834", n_epochs=n_ep)
    runs = [cifar, tiny]

    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), facecolor=SURFACE)
    ax_acc, ax_gap, ax_lc, ax_lt = axes.flat

    # A. Validation accuracy with each dataset's random-guess line
    ends = []
    for r in runs:
        h, s, c = r["h"], r["s"], r["color"]
        e = list(range(1, len(h["val_acc"]) + 1))
        ax_acc.plot(e, h["val_acc"], color=c, linewidth=1.8, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax_acc.axhline(s["chance"], color=c, linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
        ax_acc.annotate(f"chance {s['chance']:g}%", xy=(n_ep + 3.5, s["chance"]), xytext=(0, 3),
                        textcoords="offset points", fontsize=8, color=MUTED, va="bottom", ha="right")
        ends.append((e[-1], h["val_acc"][-1], r["name"]))
    style_axis(ax_acc, "Validation accuracy vs. random guessing", "Top-1 accuracy (%)")
    epoch_ticks(ax_acc, n_ep, right_pad=3.6)
    ax_acc.set_ylim(bottom=0)
    lo, hi = ax_acc.get_ylim()
    direct_labels(ax_acc, ends, min_gap=(hi - lo) * 0.06)

    # B. Generalization gap (clean train accuracy if recorded)
    ends = []
    use_clean = all(r["h"].get("train_acc_clean") for r in runs)
    for r in runs:
        h, c = r["h"], r["color"]
        e = list(range(1, len(h["val_acc"]) + 1))
        train = h["train_acc_clean"] if use_clean else h["train_acc"]
        gap = np.subtract(train, h["val_acc"])
        ax_gap.plot(e, gap, color=c, linewidth=1.8, linestyle="--" if use_clean else ":", marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ends.append((e[-1], gap[-1], r["name"]))
    ax_gap.axhline(0, color=AXIS, linewidth=1.0, zorder=2)
    style_axis(ax_gap, "Generalization gap  Δ = train acc − val acc", "Percentage points")
    epoch_ticks(ax_gap, n_ep, right_pad=3.6)
    lo, hi = ax_gap.get_ylim()
    direct_labels(ax_gap, ends, min_gap=(hi - lo) * 0.06)
    ax_gap.annotate("train acc measured on un-augmented training images, eval mode" if use_clean
                    else "train acc as logged during training", xy=(0.97, 0.04), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8, color=MUTED)

    # C/D. Loss curves, one panel per dataset (different scales: chance-level loss is ln 10 vs ln 200)
    loss_panel(ax_lc, cifar, n_ep)
    loss_panel(ax_lt, tiny, n_ep)

    handles = [Line2D([], [], color=r["color"], linewidth=2.2) for r in runs]
    labels = [r["name"] for r in runs]
    handles += [Line2D([], [], color=INK_2, linewidth=1.8, marker="o", markersize=4.5, markeredgecolor=SURFACE),
                Line2D([], [], color=INK_2, linewidth=1.4, linestyle=":"),
                Line2D([], [], color=INK_2, linewidth=1.0, linestyle="--")]
    labels += ["Validation", "Train (logged)", "Random-guess accuracy"]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.955), labelcolor=INK, handlelength=2.6, columnspacing=1.6)
    fig.suptitle(f"AlexNet-64 + BatchNorm, CIFAR-10 vs. Tiny ImageNet-200 "
                 f"({n_ep} epochs, augmentation on, batch {tiny['h']['batch_size']}, Adam lr={tiny['h']['lr']})",
                 x=0.5, y=0.99, fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=2.5, w_pad=2.5)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved figure: {args.out}")

    # ---------------- Summary table ----------------
    def f(x, fmt="{:.2f}"):
        return "n/a" if x is None else fmt.format(x)

    header = ["Dataset", "Classes", "Train imgs/class", "Native size", "Chance acc %", "Best val acc % (epoch)",
              "Macro F1 % @ best", "Final val acc %", "Final macro F1 %", "Final acc / chance", "Final train acc % (clean)",
              "Gap (clean)", "Gap (logged)", "Start loss ln(K)", "Final train loss", "Final val loss",
              "Min val loss (epoch)", "Mean s/epoch"]
    rows = []
    for r in runs:
        s = r["s"]
        rows.append([
            r["name"], str(s["classes"]), f"{s['per_class']:,}", s["res"], f"{s['chance']:g}",
            f"{s['best_val_acc']:.2f} ({s['best_epoch']})", f(s["f1_at_best"]), f(s["val_acc"]), f(s["val_f1"]),
            f"{s['val_acc'] / s['chance']:.1f}x", f(s["train_clean"]), f(s["gap_clean"]), f(s["gap_logged"]),
            f(s["init_loss"], "{:.3f}"), f(s["train_loss"], "{:.4f}"), f(s["val_loss"], "{:.4f}"),
            f"{s['min_val_loss']:.4f} ({s['min_val_loss_epoch']})", f(s["mean_epoch_s"], "{:.1f}"),
        ])
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print(f"\nCIFAR-10 numbers are the first {n_ep} epochs of {args.cifar}.")

    csv_path = os.path.splitext(args.out)[0] + "_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerows([header] + rows)
    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()
