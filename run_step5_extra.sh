#!/bin/bash
# ELEC 475 Lab 2 - Step 5 extras: multi-seed runs and the extended ablation exploration.
# Same protocol as the master tables (10 epochs, Adam, weight decay 1e-4) unless a run changes one setting.
# Finished runs (history file present) are skipped, so the script can be restarted.
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
caffeinate -i -w $$ &
run() {  # run <name> <extra train.py args...>
  local name=$1; shift
  if [ -f "logs/$name.json" ]; then echo "== $name: already done"; return; fi
  echo "== $(date +%H:%M) starting $name"
  $PY train.py --model alexnet --epochs 10 --history-file "logs/$name.json" --plot-file "plots/loss_$name.png" "$@" > "logs/$name.log" 2>&1
  echo "== $(date +%H:%M) finished $name: $(grep 'Best Val' logs/$name.log)"
}
# Multi-seed (seed 42 runs already exist: alexnet_bn.json, alexnet_with_aug.json)
run seed1_bn      --norm bn --dataset cifar10 --no-augment --seed 1
run seed2_bn      --norm bn --dataset cifar10 --no-augment --seed 2
run seed1_bn_aug  --norm bn --dataset cifar10 --seed 1
run seed2_bn_aug  --norm bn --dataset cifar10 --seed 2
# Exploration
run explore_none_tiny_lr1e-4 --norm none --dataset tiny_imagenet --no-augment --lr 1e-4
run explore_bn_aug_lr3e-3    --norm bn --dataset cifar10 --lr 3e-3
run explore_bn_aug_bs256     --norm bn --dataset cifar10 --batch-size 256
echo "== $(date +%H:%M) all extra runs done"
