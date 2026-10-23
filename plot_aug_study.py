"""
ELEC 475: Lab 2 - Step 3 Data Augmentation Study (AlexNet + BatchNorm, with vs. without augmentation)

Reads the --history-file JSONs written by train.py and produces:
  - plots/aug_study.png          : one comparison figure for the report
  - plots/aug_study_summary.csv  : the numbers behind it (also printed as a Markdown table)

Generalization gap: Gap = train accuracy - validation accuracy, reported two ways:
  - logged : the running train accuracy train.py prints. It is measured on the images the model trains on
             (augmented in the augmented run) with dropout active, so it understates how well the model fits
             its training set - more so in the augmented run.
  - clean  : accuracy on a fixed subset of un-augmented training images in eval mode, i.e. measured the
             same way as validation (needs train.py --clean-train-eval N).

Usage:
    python plot_aug_study.py                      # reads logs/alexnet_{no_aug,with_aug}.json
    python plot_aug_study.py --log-dir logs --out plots/aug_study.png
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_norm_study import AXIS, GRID, INK, INK_2, MUTED, SURFACE, direct_labels, style_axis

RUNS = [  # (file key, display name, series color) - fixed order; color follows the run
    ("no_aug", "No augmentation", "#2a78d6"),
    ("with_aug", "With augmentation", "#eb6834"),
]


def clean_train(h):
    return h.get("train_acc_clean") or None


def summarize(h):
    best = int(np.argmax(h["val_acc"]))  # train.py saves the checkpoint with the best val accuracy
    clean = clean_train(h)
    min_vl = int(np.argmin(h["val_loss"]))
    return {
        "epochs": len(h["val_acc"]),
        "train_logged": h["train_acc"][-1],
        "train_clean": clean[-1] if clean else None,
        "val_acc": h["val_acc"][-1],
        "val_f1": h["val_f1"][-1],
        "gap_logged": h["train_acc"][-1] - h["val_acc"][-1],
        "gap_clean": clean[-1] - h["val_acc"][-1] if clean else None,
        "best_val_acc": h["val_acc"][best],
        "best_epoch": best + 1,
        "f1_at_best": h["val_f1"][best],
        "gap_clean_at_best": clean[best] - h["val_acc"][best] if clean else None,
        "final_val_loss": h["val_loss"][-1],
        "min_val_loss": h["val_loss"][min_vl],
        "min_val_loss_epoch": min_vl + 1,
        "final_train_loss": h["train_loss"][-1],
        "mean_epoch_s": float(np.mean(h["epoch_time_s"])),
        "total_min": h["total_time_s"] / 60,
    }


def epoch_ticks(ax, n, right_pad):
    ax.set_xlabel("Epoch", fontsize=9, color=INK_2)
    ax.set_xticks(range(1, n + 1, 2 if n > 10 else 1))
    ax.set_xlim(0.5, n + right_pad)


def accuracy_panel(ax, h, name, color, n_ep):
    """Train vs validation accuracy for one run; the shaded band is the generalization gap."""
    e = list(range(1, len(h["val_acc"]) + 1))
    clean = clean_train(h)
    ref = clean if clean else h["train_acc"]
    ax.fill_between(e, h["val_acc"], ref, color=color, alpha=0.13, linewidth=0, zorder=1)
    ax.plot(e, h["train_acc"], color=color, linewidth=1.4, linestyle=":", zorder=3)
    if clean:
        ax.plot(e, clean, color=color, linewidth=1.8, linestyle="--", zorder=3)
    ax.plot(e, h["val_acc"], color=color, linewidth=1.8, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
    style_axis(ax, name, "Top-1 accuracy (%)")
    epoch_ticks(ax, n_ep, right_pad=4.2)
    ends = [(e[-1], h["val_acc"][-1], "Validation"), (e[-1], h["train_acc"][-1], "Train (logged)")]
    if clean:
        ends.append((e[-1], clean[-1], "Train (clean)"))
    return ends, ref[-1] - h["val_acc"][-1]


def main():
    p = argparse.ArgumentParser(description="Plot the data augmentation study")
    p.add_argument("--log-dir", default="logs", help="Folder holding alexnet_{no_aug,with_aug}.json")
    p.add_argument("--out", default="plots/aug_study.png")
    args = p.parse_args()

    runs = []
    for key, name, color in RUNS:
        path = os.path.join(args.log_dir, f"alexnet_{key}.json")
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found")
            continue
        with open(path) as f:
            h = json.load(f)
        runs.append((key, name, color, h, summarize(h)))
    if not runs:
        raise SystemExit("No history files found. Run train.py with --history-file first.")
    has_clean = all(clean_train(h) for *_, h, _ in runs)

    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), facecolor=SURFACE)
    n_ep = max(len(h["val_acc"]) for *_, h, _ in runs)

    # A/B. Train vs validation accuracy per run, same y-range so the gaps compare directly
    lo = min(min(h["val_acc"] + h["train_acc"] + (clean_train(h) or [])) for *_, h, _ in runs)
    hi = max(max(h["val_acc"] + h["train_acc"] + (clean_train(h) or [])) for *_, h, _ in runs)
    ylim = (np.floor(lo / 5) * 5, np.ceil(hi / 5) * 5 + 2)
    for ax, (_, name, color, h, s) in zip(axes[0], runs):
        ends, gap = accuracy_panel(ax, h, name, color, n_ep)
        ax.set_ylim(*ylim)
        direct_labels(ax, ends, min_gap=(ylim[1] - ylim[0]) * 0.055)
        ax.annotate(f"Gap at epoch {len(h['val_acc'])}: {gap:.1f} pts", xy=(0.03, 0.96), xycoords="axes fraction",
                    ha="left", va="top", fontsize=9, color=INK, fontweight="bold")
    for ax in axes[0][len(runs):]:
        ax.set_visible(False)

    # C. Generalization gap per epoch
    ax_gap = axes[1][0]
    ends = []
    for _, name, color, h, _ in runs:
        e = list(range(1, len(h["val_acc"]) + 1))
        logged = np.subtract(h["train_acc"], h["val_acc"])
        ax_gap.plot(e, logged, color=color, linewidth=1.4, linestyle=":", zorder=3)
        if clean_train(h):
            clean = np.subtract(clean_train(h), h["val_acc"])
            ax_gap.plot(e, clean, color=color, linewidth=1.8, linestyle="--", marker="o", markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
            ends.append((e[-1], clean[-1], name))
        else:
            ends.append((e[-1], logged[-1], name))
    ax_gap.axhline(0, color=AXIS, linewidth=1.0, zorder=2)
    style_axis(ax_gap, "Generalization gap  Δ = train acc − val acc", "Percentage points")
    epoch_ticks(ax_gap, n_ep, right_pad=4.2)
    g_lo, g_hi = ax_gap.get_ylim()
    direct_labels(ax_gap, ends, min_gap=(g_hi - g_lo) * 0.06)

    # D. Validation accuracy, ring on the saved (best) epoch
    ax_va = axes[1][1]
    ends = []
    for _, name, color, h, s in runs:
        e = list(range(1, len(h["val_acc"]) + 1))
        ax_va.plot(e, h["val_acc"], color=color, linewidth=1.8, marker="o", markersize=4.5,
                   markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax_va.plot(s["best_epoch"], s["best_val_acc"], marker="o", markersize=11, markerfacecolor="none",
                   markeredgecolor=color, markeredgewidth=1.6, zorder=4)
        ends.append((e[-1], h["val_acc"][-1], name))
    style_axis(ax_va, "Validation accuracy", "Top-1 accuracy (%)")
    epoch_ticks(ax_va, n_ep, right_pad=4.2)
    v_lo, v_hi = ax_va.get_ylim()
    direct_labels(ax_va, ends, min_gap=(v_hi - v_lo) * 0.06)
    ax_va.annotate("○ = best epoch (saved checkpoint)", xy=(0.03, 0.96), xycoords="axes fraction",
                   ha="left", va="top", fontsize=8, color=MUTED)

    # Legend: run colors + line-style key
    handles = [Line2D([], [], color=c, linewidth=2.2) for _, _, c, _, _ in runs]
    labels = [n for _, n, _, _, _ in runs]
    handles += [Line2D([], [], color=INK_2, linewidth=1.8, marker="o", markersize=4.5, markeredgecolor=SURFACE)]
    labels += ["Validation"]
    if has_clean:
        handles += [Line2D([], [], color=INK_2, linewidth=1.8, linestyle="--")]
        labels += ["Train, clean (un-augmented, eval mode)"]
    handles += [Line2D([], [], color=INK_2, linewidth=1.4, linestyle=":")]
    labels += ["Train, logged (as trained, dropout on)"]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8.8,
               bbox_to_anchor=(0.5, 0.955), labelcolor=INK, handlelength=2.6, columnspacing=1.4)
    first = runs[0][3]
    fig.suptitle(f"AlexNet-64 + BatchNorm on CIFAR-10: effect of data augmentation "
                 f"({n_ep} epochs, batch {first['batch_size']}, Adam lr={first['lr']})",
                 x=0.5, y=0.99, fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=2.5, w_pad=2.5)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved figure: {args.out}")

    # ---------------- Summary table ----------------
    def f(x, fmt="{:.2f}"):
        return "n/a" if x is None else fmt.format(x)

    header = ["Run", "Final train acc % (logged)", "Final train acc % (clean)", "Final val acc %",
              "Gap (logged)", "Gap (clean)", "Final macro F1 %", "Best val acc % (epoch)", "Macro F1 % @ best",
              "Gap (clean) @ best", "Final val loss", "Min val loss (epoch)", "Final train loss", "Mean s/epoch",
              "Total min"]
    rows = []
    for _, name, _, _, s in runs:
        rows.append([
            name, f(s["train_logged"]), f(s["train_clean"]), f(s["val_acc"]), f(s["gap_logged"]),
            f(s["gap_clean"]), f(s["val_f1"]), f"{s['best_val_acc']:.2f} ({s['best_epoch']})", f(s["f1_at_best"]),
            f(s["gap_clean_at_best"]), f(s["final_val_loss"], "{:.4f}"),
            f"{s['min_val_loss']:.4f} ({s['min_val_loss_epoch']})", f(s["final_train_loss"], "{:.4f}"),
            f(s["mean_epoch_s"], "{:.1f}"), f(s["total_min"], "{:.1f}"),
        ])
    if len(runs) == 2:
        a, b = runs[0][4], runs[1][4]
        d = lambda k: None if a[k] is None or b[k] is None else b[k] - a[k]
        rows.append([
            "Change (with - without)", f(d("train_logged"), "{:+.2f}"), f(d("train_clean"), "{:+.2f}"),
            f(d("val_acc"), "{:+.2f}"), f(d("gap_logged"), "{:+.2f}"), f(d("gap_clean"), "{:+.2f}"),
            f(d("val_f1"), "{:+.2f}"), f(d("best_val_acc"), "{:+.2f}"), f(d("f1_at_best"), "{:+.2f}"),
            f(d("gap_clean_at_best"), "{:+.2f}"), f(d("final_val_loss"), "{:+.4f}"), f(d("min_val_loss"), "{:+.4f}"),
            f(d("final_train_loss"), "{:+.4f}"), f(d("mean_epoch_s"), "{:+.1f}"), f(d("total_min"), "{:+.1f}"),
        ])
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print("\nGap = train acc - val acc (percentage points). 'logged' uses train.py's running train accuracy "
          "(on augmented images, dropout on); 'clean' uses un-augmented training images in eval mode.")

    csv_path = os.path.splitext(args.out)[0] + "_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerows([header] + rows)
    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()
