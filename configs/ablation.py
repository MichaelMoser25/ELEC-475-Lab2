"""
ELEC 475: Lab 2 - Step 5 Master Ablation Ladder

One entry per row of master tables 7.3a / 7.3b. Each row names the train.py history file (--history-file)
of its CIFAR-10 and Tiny ImageNet runs. Every run uses batch 128, Adam (lr 1e-3, weight decay 1e-4), seed 42,
and the tables read the first ABLATION_EPOCHS epochs of each run.

Used by make_master_tables.py and `python -m utils.visualization --plot-ablation`; run_step5.sh trains the
runs that are still missing.
"""

ABLATION_EPOCHS = 10

ABLATION_RUNS = [
    {"label": "1. Base LeNet-5", "short": "LeNet-5", "model": "lenet", "norm": "none", "augment": False,
     "conv_depth": 2, "regularization": "None", "augmentation": "None",
     "cifar10": "logs/lenet_cifar10.json", "tiny_imagenet": "logs/lenet_tiny.json"},
    {"label": "2. Modernized LeNet", "short": "Modern LeNet", "model": "lenet_modern", "norm": "none", "augment": False,
     "conv_depth": 2, "regularization": "None (ReLU + overlapping MaxPool)", "augmentation": "None",
     "cifar10": "logs/lenet_modern_cifar10.json", "tiny_imagenet": "logs/lenet_modern_tiny.json"},
    {"label": "3. AlexNet (No Norm)", "short": "AlexNet", "model": "alexnet", "norm": "none", "augment": False,
     "conv_depth": 5, "regularization": "Dropout (p = 0.5)", "augmentation": "None",
     "cifar10": "logs/alexnet_none.json", "tiny_imagenet": "logs/alexnet_none_tiny.json"},
    {"label": "4. AlexNet + LRN", "short": "AlexNet + LRN", "model": "alexnet", "norm": "lrn", "augment": False,
     "conv_depth": 5, "regularization": "LRN + Dropout (p = 0.5)", "augmentation": "None",
     "cifar10": "logs/alexnet_lrn.json", "tiny_imagenet": "logs/alexnet_lrn_tiny.json"},
    {"label": "5. AlexNet + BatchNorm", "short": "AlexNet + BN", "model": "alexnet", "norm": "bn", "augment": False,
     "conv_depth": 5, "regularization": "BatchNorm2d + Dropout (p = 0.5)", "augmentation": "None",
     "cifar10": "logs/alexnet_bn.json", "tiny_imagenet": "logs/alexnet_bn_tiny.json"},
    {"label": "6. AlexNet + BN + Aug", "short": "AlexNet + BN + Aug", "model": "alexnet", "norm": "bn", "augment": True,
     "conv_depth": 5, "regularization": "BatchNorm2d + Dropout (p = 0.5)", "augmentation": "Pure PyTorch Aug",
     "cifar10": "logs/alexnet_with_aug.json", "tiny_imagenet": "logs/alexnet_tiny_imagenet.json"},
    {"label": "7. PyTorch Zoo AlexNet", "short": "Zoo AlexNet", "model": "zoo_alexnet", "norm": "none", "augment": False,
     "conv_depth": 5, "regularization": "PyTorch Zoo (AdaptiveAvgPool2d) + Dropout (p = 0.5)", "augmentation": "None",
     "cifar10": "logs/zoo_alexnet_cifar10.json", "tiny_imagenet": "logs/zoo_alexnet_tiny.json"},
]
