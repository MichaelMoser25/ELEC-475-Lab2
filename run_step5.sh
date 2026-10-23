#!/bin/bash
# ELEC 475 Lab 2 - Step 5: train every run the master tables (7.3a/7.3b) still need, then build the
# tables and the ablation plots. Each run: 10 epochs, no augmentation (row 6's runs already exist).
# A run whose history file exists is skipped, so after an interruption just start this script again.
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
caffeinate -i -w $$ &   # keep the Mac from sleeping until this script ends (keep the charger plugged in)

run() {  # run <name> <model> <norm> <dataset>
  local name=$1 model=$2 norm=$3 dataset=$4
  if [ -f "logs/$name.json" ]; then echo "== $name: already done, skipping"; return; fi
  echo "== $(date +%H:%M) starting $name"
  $PY train.py --model "$model" --norm "$norm" --dataset "$dataset" --epochs 10 --no-augment \
    --save-model "checkpoints/$name.pth" --plot-file "plots/loss_$name.png" \
    --history-file "logs/$name.json" 2>&1 | tee "logs/$name.log"
}

# CIFAR-10 (about 35 min)
run lenet_cifar10        lenet        none cifar10
run lenet_modern_cifar10 lenet_modern none cifar10
run zoo_alexnet_cifar10  zoo_alexnet  none cifar10

# Tiny ImageNet (about 3 h; the slow LRN run goes last)
run lenet_tiny           lenet        none tiny_imagenet
run lenet_modern_tiny    lenet_modern none tiny_imagenet
run alexnet_none_tiny    alexnet      none tiny_imagenet
run alexnet_bn_tiny      alexnet      bn   tiny_imagenet
run zoo_alexnet_tiny     zoo_alexnet  none tiny_imagenet
run alexnet_lrn_tiny     alexnet      lrn  tiny_imagenet

echo "== $(date +%H:%M) training finished; building tables and plots"
$PY compare_models.py --dataset cifar10 --device mps --reps 200 2>&1 | tee logs/compare_models_cifar10.txt
$PY -m utils.visualization --plot-ablation
$PY make_master_tables.py
