"""
ELEC 475: Lab 2 - Side-by-side training/validation curves for every master-ablation row

One figure per dataset: a row per model (configs/ablation.py), loss on the left and accuracy on the right,
training dotted and validation solid. Shared y-axes per column, so rows compare directly.

Usage:
    python plot_training_grid.py            # writes plots/training_grid_cifar10.png and _tiny_imagenet.png
"""
import argparse
import json
import math
import os

import matplotlib.pyplot as plt

from configs.ablation import ABLATION_EPOCHS, ABLATION_RUNS
from plot_norm_study import INK, INK_2, MUTED, SURFACE, style_axis

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']  # same slots as the ablation plot
NAMES = {"cifar10": ("CIFAR-10", 10), "tiny_imagenet": ("Tiny ImageNet-200", 200)}


def main():
    p = argparse.ArgumentParser(description="Training/validation curves for every ablation row")
    p.add_argument("--dataset", default="both", choices=["cifar10", "tiny_imagenet", "both"])
    p.add_argument("--out-dir", default="plots")
    args = p.parse_args()
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    for ds in (["cifar10", "tiny_imagenet"] if args.dataset == "both" else [args.dataset]):
        name, k = NAMES[ds]
        rows = []
        for slot, r in enumerate(ABLATION_RUNS):
            path = r[ds]
            h = None
            if os.path.exists(path):
                with open(path) as f:
                    h = json.load(f)
            rows.append((slot, r, h))
        done = [h for _, _, h in rows if h]
        if not done:
            print(f"{name}: no runs yet, skipped")
            continue
        loss_hi = max(max(h["train_loss"][:ABLATION_EPOCHS] + h["val_loss"][:ABLATION_EPOCHS]) for h in done)
        acc_hi = max(max(h["train_acc"][:ABLATION_EPOCHS] + h["val_acc"][:ABLATION_EPOCHS]) for h in done)

        fig, axes = plt.subplots(len(rows), 2, figsize=(9, 1.8 * len(rows) + 0.6), facecolor=SURFACE,
                                 sharex=True, squeeze=False)
        for (slot, r, h), (ax_l, ax_a) in zip(rows, axes):
            c = COLORS[slot % len(COLORS)]
            for ax, ylabel in ((ax_l, "Cross-entropy"), (ax_a, "Top-1 (%)")):
                style_axis(ax, "", ylabel)
                ax.set_xticks(range(1, ABLATION_EPOCHS + 1))
            ax_l.set_ylim(0, max(loss_hi, math.log(k)) * 1.08)
            ax_a.set_ylim(0, min(100, acc_hi * 1.12))
            ax_l.set_title(f"{r['label']}: loss", loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=4)
            ax_a.set_title(f"{r['label']}: accuracy", loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=4)
            if h is None:
                for ax in (ax_l, ax_a):
                    ax.annotate("pending", xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
                                fontsize=9, color=MUTED)
                continue
            e = list(range(1, min(ABLATION_EPOCHS, len(h["val_acc"])) + 1))
            n = len(e)
            ax_l.plot(e, h["train_loss"][:n], color=c, linewidth=1.4, linestyle=":")
            ax_l.plot(e, h["val_loss"][:n], color=c, linewidth=1.8, marker="o", markersize=3.5, markeredgecolor=SURFACE)
            ax_l.axhline(math.log(k), color=MUTED, linewidth=0.8, linestyle="--")
            ax_a.plot(e, h["train_acc"][:n], color=c, linewidth=1.4, linestyle=":")
            ax_a.plot(e, h["val_acc"][:n], color=c, linewidth=1.8, marker="o", markersize=3.5, markeredgecolor=SURFACE)
            best = max(range(n), key=lambda i: h["val_acc"][i])
            ax_a.annotate(f"best val {h['val_acc'][best]:.1f}% (ep {best + 1})", xy=(0.02, 0.92), xycoords="axes fraction",
                          ha="left", va="top", fontsize=8, color=INK_2)
        for ax in axes[-1]:
            ax.set_xlabel("Epoch", fontsize=9, color=INK_2)
        fig.suptitle(f"Training (dotted) and validation (solid) curves on {name}, {ABLATION_EPOCHS} epochs; "
                     f"dashed line = chance-level loss ln {k}", x=0.5, y=0.997, fontsize=10.5, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.1, w_pad=2.5)
        out = os.path.join(args.out_dir, f"training_grid_{ds}.png")
        os.makedirs(args.out_dir, exist_ok=True)
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
