"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png",
    title: str = "ELEC 475: Master Ablation Study (Validation Accuracy)"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    # Fixed-order colour-blind-safe categorical palette: a row keeps its colour in every figure
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        slot = data.get("slot", idx)
        ax.plot(epochs, val_accs, color=colors[slot % len(colors)], marker=markers[slot % len(markers)],
                markersize=5, linewidth=2, label=model_label)

    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
    n_epochs = max((len(d.get("val_accs", [])) for d in ablation_results.values()), default=1)
    ax.set_xticks(range(1, n_epochs + 1))
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Master ablation summary plot saved to: {save_path}")


def load_ablation_curves(dataset: str = "cifar10", epochs: int = None) -> Dict[str, Dict[str, List[float]]]:
    """Validation-accuracy curves of the master ablation rows (configs/ablation.py) that have a history file."""
    import json
    from configs.ablation import ABLATION_RUNS, ABLATION_EPOCHS
    epochs = epochs or ABLATION_EPOCHS
    results = {}
    for slot, row in enumerate(ABLATION_RUNS):
        path = row[dataset]
        if not os.path.exists(path):
            print(f"  missing ({row['label']}): {path}")
            continue
        with open(path) as f:
            hist = json.load(f)
        results[row["label"]] = {"val_accs": hist["val_acc"][:epochs], "slot": slot}
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ELEC 475 Lab 2 plotting utilities")
    parser.add_argument("--plot-ablation", action="store_true",
                        help="Plot validation accuracy of every master-ablation row (needs train.py --history-file logs)")
    parser.add_argument("--dataset", default="both", choices=["cifar10", "tiny_imagenet", "both"])
    parser.add_argument("--out-dir", default="./plots")
    args = parser.parse_args()
    if not args.plot_ablation:
        parser.error("nothing to do: pass --plot-ablation")
    names = {"cifar10": "CIFAR-10", "tiny_imagenet": "Tiny ImageNet-200"}
    for ds in (["cifar10", "tiny_imagenet"] if args.dataset == "both" else [args.dataset]):
        print(f"{names[ds]}:")
        curves = load_ablation_curves(ds)
        if not curves:
            print("  no history files yet, skipped")
            continue
        plot_ablation_summary(curves, save_path=os.path.join(args.out_dir, f"ablation_summary_{ds}.png"),
                              title=f"Master ablation study: validation accuracy on {names[ds]}")
