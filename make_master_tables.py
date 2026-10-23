"""
ELEC 475: Lab 2 - Step 5 Master Ablation Tables (7.3a and 7.3b)

Builds both report tables from the run histories listed in configs/ablation.py:
  - Trainable params and FLOPs: ModelProfiler (thop, 1 MAC = 2 FLOPs), 64x64 input, K = 10 classes
  - Inference latency: median of `--reps` batch-1 forward passes on `--device` (GPU-synchronized)
  - Mean epoch time: mean over the first 10 epochs of each run, training + validation, as train.py times it
  - Val accuracy / macro F1: best validation accuracy within the first 10 epochs (the epoch train.py
    checkpoints) and the macro F1 of that same epoch
Rows whose history file does not exist yet show "pending".

Writes plots/master_tables.md and plots/master_table_a.csv / master_table_b.csv.

Usage:
    python make_master_tables.py
    python make_master_tables.py --device cpu --reps 300
"""
import argparse
import contextlib
import csv
import io
import json
import os
import statistics

import torch

from configs.ablation import ABLATION_EPOCHS, ABLATION_RUNS
from models import get_model
from utils.profiler import ModelProfiler, benchmark_inference


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        h = json.load(f)
    for key in ("val_acc", "val_f1", "val_loss", "train_acc", "train_loss", "epoch_time_s"):
        h[key] = h[key][:ABLATION_EPOCHS]
    return h


def best(h):
    i = max(range(len(h["val_acc"])), key=lambda k: h["val_acc"][k])
    return h["val_acc"][i], h["val_f1"][i], i + 1


def main():
    p = argparse.ArgumentParser(description="Build master tables 7.3a and 7.3b")
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    p.add_argument("--device", default=default_device, choices=["cuda", "mps", "cpu"])
    p.add_argument("--reps", type=int, default=300, help="Batch-1 forward passes per latency measurement")
    p.add_argument("--out-dir", default="plots")
    args = p.parse_args()

    complexity, latency = {}, {}
    for row in ABLATION_RUNS:
        key = (row["model"], row["norm"])
        if key in complexity:  # row 6 is the same network as row 5
            continue
        model = get_model(row["model"], num_classes=10, norm_type=row["norm"])
        with contextlib.redirect_stdout(io.StringIO()):
            prof = ModelProfiler(model, input_size=(1, 3, 64, 64), device="cpu")
        complexity[key] = (prof.trainable_params / 1e6, prof.flops / 1e6)
        runs = [benchmark_inference(model, input_size=(1, 3, 64, 64), device=args.device, reps=args.reps)[0]
                for _ in range(3)]
        latency[key] = statistics.median(runs)

    a_header = ["Model Architecture", "Conv Depth", "Normalization & Regularization", "Data Augmentation",
                "Trainable Params (M)", "FLOPs (MFLOPs)", "Mean Epoch Time (s), CIFAR-10 / Tiny ImageNet",
                f"Inference Latency (ms), {args.device.upper()} batch 1"]
    b_header = ["Model Architecture", "CIFAR-10 Val Acc (%)", "CIFAR-10 Val Macro F1 (%)",
                "Tiny ImageNet Val Acc (%)", "Tiny ImageNet Val Macro F1 (%)"]
    a_rows, b_rows, notes = [], [], []
    for row in ABLATION_RUNS:
        key = (row["model"], row["norm"])
        params, flops = complexity[key]
        hc, ht = load(row["cifar10"]), load(row["tiny_imagenet"])
        t = lambda h: f"{statistics.mean(h['epoch_time_s']):.1f}" if h else "pending"
        a_rows.append([row["label"], str(row["conv_depth"]), row["regularization"], row["augmentation"],
                       f"{params:.3f}", f"{flops:.2f}", f"{t(hc)} / {t(ht)}", f"{latency[key]:.2f}"])
        cells = [row["label"]]
        for name, h in (("CIFAR-10", hc), ("Tiny ImageNet", ht)):
            if h is None:
                cells += ["pending", "pending"]
                continue
            acc, f1, ep = best(h)
            cells += [f"{acc:.2f}", f"{f1:.2f}"]
            n = len(h["val_acc"])
            if n < ABLATION_EPOCHS:
                notes.append(f"{row['label']} on {name}: only {n} epochs recorded.")
            notes.append(f"{row['label']} on {name}: best epoch {ep} of {n}.")
        b_rows.append(cells)

    def md(header, rows):
        return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                         ["| " + " | ".join(r) + " |" for r in rows])

    text = (f"## Table 7.3a: Architectural configuration and computational complexity\n\n{md(a_header, a_rows)}\n\n"
            f"Params and FLOPs for 64x64 inputs and 10 classes (thop: 1 MAC = 2 FLOPs). Epoch time = training + "
            f"validation, mean of the first {ABLATION_EPOCHS} epochs. Latency = median of {args.reps} batch-1 "
            f"forward passes on {args.device.upper()}, median of 3 repeats.\n\n"
            f"## Table 7.3b: Empirical predictive performance across datasets\n\n{md(b_header, b_rows)}\n\n"
            f"Best validation accuracy within the first {ABLATION_EPOCHS} epochs, with the macro F1 of that epoch.\n\n"
            "Best epochs:\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print(text)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "master_tables.md"), "w") as f:
        f.write(text)
    for name, header, rows in (("master_table_a.csv", a_header, a_rows), ("master_table_b.csv", b_header, b_rows)):
        with open(os.path.join(args.out_dir, name), "w", newline="") as f:
            csv.writer(f).writerows([header] + rows)
    print(f"Saved {args.out_dir}/master_tables.md, master_table_a.csv, master_table_b.csv")


if __name__ == "__main__":
    main()
