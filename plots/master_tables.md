## Table 7.3a: Architectural configuration and computational complexity

| Model Architecture | Conv Depth | Normalization & Regularization | Data Augmentation | Trainable Params (M) | FLOPs (MFLOPs) | Mean Epoch Time (s), CIFAR-10 / Tiny ImageNet | Inference Latency (ms), MPS batch 1 |
|---|---|---|---|---|---|---|---|
| 1. Base LeNet-5 | 2 | None | None | 0.338 | 7.17 | 34.2 / 43.7 | 0.58 |
| 2. Modernized LeNet | 2 | None (ReLU + overlapping MaxPool) | None | 0.446 | 9.19 | 36.5 / 45.4 | 0.55 |
| 3. AlexNet (No Norm) | 5 | Dropout (p = 0.5) | None | 6.609 | 436.70 | 71.6 / 116.5 | 1.33 |
| 4. AlexNet + LRN | 5 | LRN + Dropout (p = 0.5) | None | 6.609 | 436.70 | 208.8 / 377.3 | 2.19 |
| 5. AlexNet + BatchNorm | 5 | BatchNorm2d + Dropout (p = 0.5) | None | 6.611 | 437.97 | 76.8 / 126.6 | 3.91 |
| 6. AlexNet + BN + Aug | 5 | BatchNorm2d + Dropout (p = 0.5) | Pure PyTorch Aug | 6.611 | 437.97 | 80.0 / 129.0 | 3.91 |
| 7. PyTorch Zoo AlexNet | 5 | PyTorch Zoo (AdaptiveAvgPool2d) + Dropout (p = 0.5) | None | 57.045 | 188.20 | 74.8 / 140.8 | 3.41 |

Params and FLOPs for 64x64 inputs and 10 classes (thop: 1 MAC = 2 FLOPs). Epoch time = training + validation, mean of the first 10 epochs. Latency = median of 300 batch-1 forward passes on MPS, median of 3 repeats.

## Table 7.3b: Empirical predictive performance across datasets

| Model Architecture | CIFAR-10 Val Acc (%) | CIFAR-10 Val Macro F1 (%) | Tiny ImageNet Val Acc (%) | Tiny ImageNet Val Macro F1 (%) |
|---|---|---|---|---|
| 1. Base LeNet-5 | 55.63 | 55.24 | 12.67 | 11.08 |
| 2. Modernized LeNet | 66.68 | 66.85 | 20.56 | 18.85 |
| 3. AlexNet (No Norm) | 74.30 | 73.85 | 0.50 | 0.00 |
| 4. AlexNet + LRN | 74.27 | 74.30 | 15.21 | 12.50 |
| 5. AlexNet + BatchNorm | 79.19 | 79.01 | 29.30 | 27.55 |
| 6. AlexNet + BN + Aug | 80.27 | 80.12 | 28.30 | 26.81 |
| 7. PyTorch Zoo AlexNet | 66.76 | 66.86 | 0.50 | 0.00 |

Best validation accuracy within the first 10 epochs, with the macro F1 of that epoch.

Best epochs:
- 1. Base LeNet-5 on CIFAR-10: best epoch 9 of 10.
- 1. Base LeNet-5 on Tiny ImageNet: best epoch 10 of 10.
- 2. Modernized LeNet on CIFAR-10: best epoch 10 of 10.
- 2. Modernized LeNet on Tiny ImageNet: best epoch 9 of 10.
- 3. AlexNet (No Norm) on CIFAR-10: best epoch 10 of 10.
- 3. AlexNet (No Norm) on Tiny ImageNet: best epoch 1 of 10.
- 4. AlexNet + LRN on CIFAR-10: best epoch 10 of 10.
- 4. AlexNet + LRN on Tiny ImageNet: best epoch 10 of 10.
- 5. AlexNet + BatchNorm on CIFAR-10: best epoch 9 of 10.
- 5. AlexNet + BatchNorm on Tiny ImageNet: best epoch 10 of 10.
- 6. AlexNet + BN + Aug on CIFAR-10: best epoch 10 of 10.
- 6. AlexNet + BN + Aug on Tiny ImageNet: best epoch 10 of 10.
- 7. PyTorch Zoo AlexNet on CIFAR-10: best epoch 10 of 10.
- 7. PyTorch Zoo AlexNet on Tiny ImageNet: best epoch 1 of 10.
