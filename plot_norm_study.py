"""
ELEC 475: Lab 2 - Step 2 Normalization Study (None vs. LRN vs. BatchNorm2d)

Reads the --history-file JSONs written by train.py for the three AlexNet variants and produces:
  - plots/norm_study.png          : one comparison figure for the report
  - plots/norm_study_summary.csv  : the numbers behind it (also printed as a Markdown table)

The figure and table map onto the report questions:
  Performance   -> validation accuracy curve, best/final val accuracy and Macro F1
  Overhead      -> FLOPs per image and wall-clock time per epoch, relative to no normalization
  Stability     -> per-iteration training loss, batch-loss noise, number of val-loss increases

Usage:
    python plot_norm_study.py                      # reads logs/alexnet_{none,lrn,bn}.json
    python plot_norm_study.py --log-dir logs --out plots/norm_study.png
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RUNS = [  # (norm flag, display name, series color) - fixed order; color follows the variant
    ("none", "A: No norm", "#2a78d6"),
    ("lrn", "B: LRN (2012)", "#eb6834"),
    ("bn", "C: BatchNorm2d", "#1baf7a"),
]

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def moving_average(x, w):
    """Centered moving average; returns (values, index offset of first value)."""
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="valid"), (w - 1) / 2


def summarize(h, window):
    epochs = len(h["val_acc"])
    best_i = int(np.argmax(h["val_acc"]))  # train.py saves the checkpoint with the best val accuracy
    raw = np.asarray(h["batch_train_loss"])
    smooth, off = moving_average(raw, window)
    resid = raw[int(off): int(off) + len(smooth)] - smooth
    # Skip epoch 1 (the initial steep drop dominates any noise measure)
    start = max(0, h["batches_per_epoch"] - int(off))
    noise = float(np.std(resid[start:])) if len(resid) > start else float(np.std(resid))
    val_rises = int(sum(b > a for a, b in zip(h["val_loss"], h["val_loss"][1:])))
    return {
        "epochs": epochs,
        "best_val_acc": h["val_acc"][best_i],
        "best_epoch": best_i + 1,
        "f1_at_best": h["val_f1"][best_i],
        "final_val_acc": h["val_acc"][-1],
        "final_f1": h["val_f1"][-1],
        "final_train_loss": h["train_loss"][-1],
        "final_gap": h["train_acc"][-1] - h["val_acc"][-1],
        "noise": noise,
        "val_rises": val_rises,
        "mean_epoch_s": float(np.mean(h["epoch_time_s"])),
        "total_min": h["total_time_s"] / 60,
        "mflops": h["flops"] / 1e6,
        "params": h["params"],
    }


def style_axis(ax, title, ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def epoch_axis(ax, n_epochs, right_pad=0.5):
    ax.set_xlabel("Epoch", fontsize=9, color=INK_2)
    ax.set_xticks(range(1, n_epochs + 1))
    ax.set_xlim(0.5 if right_pad == 0.5 else 0, n_epochs + right_pad)


def direct_labels(ax, ends, min_gap):
    """Name each line at its last point (text in ink, not series color); nudge apart if crowded."""
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (x, _, name), y in zip(ends, ys):
        ax.annotate(name, xy=(x, y), xytext=(11, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)


def main():
    p = argparse.ArgumentParser(description="Plot the AlexNet normalization study")
    p.add_argument("--log-dir", default="logs", help="Folder holding alexnet_{none,lrn,bn}.json")
    p.add_argument("--out", default="plots/norm_study.png")
    p.add_argument("--window", type=int, default=25, help="Moving-average window (batches) for the loss curve")
    args = p.parse_args()

    runs = []
    for norm, name, color in RUNS:
        path = os.path.join(args.log_dir, f"alexnet_{norm}.json")
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found")
            continue
        with open(path) as f:
            h = json.load(f)
        runs.append((norm, name, color, h, summarize(h, args.window)))
    if not runs:
        raise SystemExit("No history files found. Run train.py with --history-file first.")

    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), facecolor=SURFACE)
    ax_it, ax_vl, ax_va, ax_cost = axes.flat
    n_ep = max(len(h["val_acc"]) for *_, h, _ in runs)
    ep = lambda h: list(range(1, len(h["val_acc"]) + 1))

    # A. Training loss per iteration: faint raw batches + moving average
    for _, name, color, h, _ in runs:
        raw = np.asarray(h["batch_train_loss"])
        x = np.arange(1, len(raw) + 1) / h["batches_per_epoch"]
        smooth, off = moving_average(raw, args.window)
        ax_it.plot(x, raw, color=color, linewidth=0.6, alpha=0.18, zorder=2)
        ax_it.plot(x[int(off): int(off) + len(smooth)], smooth, color=color, linewidth=1.8,
                   label=name, zorder=3)
    style_axis(ax_it, "Training loss per iteration", "Cross-entropy")
    epoch_axis(ax_it, n_ep, right_pad=0.0)
    ax_it.annotate(f"line = {args.window}-batch moving average, haze = raw batches",
                   xy=(0.99, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=8, color=MUTED)

    # B. Validation loss per epoch
    for _, name, color, h, _ in runs:
        ax_vl.plot(ep(h), h["val_loss"], color=color, linewidth=1.8, marker="o", markersize=4.5,
                   markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
    style_axis(ax_vl, "Validation loss", "Cross-entropy")
    epoch_axis(ax_vl, n_ep)

    # C. Validation accuracy per epoch, ring on the saved (best) epoch, direct labels
    for _, name, color, h, s in runs:
        ax_va.plot(ep(h), h["val_acc"], color=color, linewidth=1.8, marker="o", markersize=4.5,
                   markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax_va.plot(s["best_epoch"], s["best_val_acc"], marker="o", markersize=11, markerfacecolor="none",
                   markeredgecolor=color, markeredgewidth=1.6, zorder=4)
    style_axis(ax_va, "Validation accuracy", "Top-1 accuracy (%)")
    epoch_axis(ax_va, n_ep, right_pad=3.2)
    lo, hi = ax_va.get_ylim()
    direct_labels(ax_va, [(len(h["val_acc"]), h["val_acc"][-1], name) for _, name, _, h, _ in runs],
                  min_gap=(hi - lo) * 0.06)
    ax_va.annotate("○ = best epoch (saved checkpoint)", xy=(0.02, 0.97), xycoords="axes fraction",
                   ha="left", va="top", fontsize=8, color=MUTED)

    # D. Cost relative to the first variant (no norm): FLOPs vs wall-clock time
    base = runs[0][4]
    groups = [("FLOPs per image", "mflops"), ("Time per epoch", "mean_epoch_s")]
    bar_h = 0.22
    for gi, (gname, key) in enumerate(groups):
        for ri, (_, name, color, _, s) in enumerate(runs):
            ratio = s[key] / base[key]
            y = gi + (ri - (len(runs) - 1) / 2) * (bar_h + 0.04)
            ax_cost.barh(y, ratio, height=bar_h, color=color, zorder=3)
            ax_cost.annotate(f"×{ratio:.2f}", xy=(ratio, y), xytext=(5, 0), textcoords="offset points",
                             va="center", fontsize=8.5, color=INK)
    style_axis(ax_cost, f"Cost relative to {runs[0][1].split(': ')[-1].lower()}")
    ax_cost.grid(False, axis="y")
    ax_cost.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax_cost.axvline(1.0, color=AXIS, linewidth=1.0, zorder=2)
    ax_cost.set_yticks(range(len(groups)))
    ax_cost.set_yticklabels([g for g, _ in groups], fontsize=9, color=INK_2)
    ax_cost.invert_yaxis()
    max_ratio = max(s[k] / base[k] for *_, s in runs for _, k in groups)
    ax_cost.set_xlim(0, max_ratio * 1.2)
    ax_cost.set_xlabel("× no-norm baseline", fontsize=9, color=INK_2)

    handles, labels = ax_it.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.955), labelcolor=INK)
    first = runs[0][3]
    fig.suptitle(f"AlexNet-64 on CIFAR-10: effect of normalization "
                 f"({n_ep} epochs, batch {first['batch_size']}, Adam lr={first['lr']})",
                 x=0.5, y=0.99, fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=2.5, w_pad=2.5)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved figure: {args.out}")

    # ---------------- Summary table ----------------
    header = ["Variant", "Best val acc % (epoch)", "Macro F1 % @ best", "Final val acc %", "Final macro F1 %",
              "Final train loss", "Train-val acc gap (final)", "Batch-loss noise (std)", "Val-loss increases",
              "Mean s/epoch", "Time vs none", "MFLOPs/img", "Params", "Total min"]
    rows = []
    for _, name, _, _, s in runs:
        rows.append([
            name,
            f"{s['best_val_acc']:.2f} ({s['best_epoch']})",
            f"{s['f1_at_best']:.2f}",
            f"{s['final_val_acc']:.2f}",
            f"{s['final_f1']:.2f}",
            f"{s['final_train_loss']:.4f}",
            f"{s['final_gap']:.2f}",
            f"{s['noise']:.4f}",
            f"{s['val_rises']} of {s['epochs'] - 1}",
            f"{s['mean_epoch_s']:.1f}",
            f"x{s['mean_epoch_s'] / base['mean_epoch_s']:.2f}",
            f"{s['mflops']:.2f}",
            f"{s['params']:,}",
            f"{s['total_min']:.1f}",
        ])
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print("\nNote: FLOPs come from thop, which counts BatchNorm2d but has no counter for LocalResponseNorm "
          "(counted as 0).")

    csv_path = os.path.splitext(args.out)[0] + "_summary.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()
