"""
ELEC 475: Lab 2 - Modernized LeNet Architecture (Starter Template)
Ablation Study 1 & 2: Incorporating ReLU and Overlapping Max Pooling.

Key Modifications from Base LeNet-5:
    1. Tanh -> ReLU (Lecture 05 Slide 4-5)
    2. 2x2 Average Pooling -> 3x3 Overlapping Max Pooling with stride=2 (Lecture 05 Slide 7)
"""
import torch
import torch.nn as nn

class LeNetModern(nn.Module):
    """
    Modernized LeNet architecture incorporating ReLU activation and
    overlapping Max Pooling (3x3 with stride 2).

    Spatial dimension arithmetic:
        Input   : (B, 3, 64, 64)
        Conv1   : (64 - 5 + 2*2)/1 + 1 = 64  --> (B, 6, 64, 64)
        ReLU    : (B, 6, 64, 64)
        MaxPool1: floor((64 - 3)/2) + 1 = 31 --> (B, 6, 31, 31)
        Conv2   : (31 - 5 + 2*2)/1 + 1 = 31  --> (B, 16, 31, 31)
        ReLU    : (B, 16, 31, 31)
        MaxPool2: floor((31 - 3)/2) + 1 = 15 --> (B, 16, 15, 15)
        Flatten : 16 * 15 * 15 = 3,600 features
        FC1     : 3,600 -> 120 (with ReLU)
        FC2     : 120 -> 84 (with ReLU)
        FC3     : 84 -> num_classes
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super(LeNetModern, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # ======================================================================
        # TODO [IN-LAB STEP 4]: Implement the Modernized Feature Extractor
        #
        # Requirements:
        #   1. Conv1: in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2
        #   2. Activation: nn.ReLU(inplace=True)  [Ablation 1]
        #   3. Overlapping MaxPool1: kernel_size=3, stride=2  [Ablation 2]
        #   4. Conv2: in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2
        #   5. Activation: nn.ReLU(inplace=True)  [Ablation 1]
        #   6. Overlapping MaxPool2: kernel_size=3, stride=2  [Ablation 2]
        # ======================================================================
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),   # (B, 6, 64, 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                            # (B, 6, 31, 31)
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),             # (B, 16, 31, 31)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                            # (B, 16, 15, 15)
        )

        # ======================================================================
        # TODO [IN-LAB STEP 4]: Implement the Modernized Classifier Head
        #
        # Requirements:
        #   1. FC1: in_features = 16 * 15 * 15 (3,600), out_features = 120
        #   2. Activation: nn.ReLU(inplace=True)
        #   3. FC2: in_features = 120, out_features = 84
        #   4. Activation: nn.ReLU(inplace=True)
        #   5. FC3 (Output): in_features = 84, out_features = num_classes
        # ======================================================================
        self.classifier = nn.Sequential(
            nn.Linear(16 * 15 * 15, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # TODO [IN-LAB STEP 4]: Implement Forward Pass
        # ======================================================================
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

if __name__ == "__main__":
    from torchinfo import summary
    model = LeNetModern(num_classes=10, in_channels=3)
    dummy_input = torch.randn(1, 3, 64, 64)
    print("Testing Modernized LeNet Architecture:")
    try:
        summary(model, input_size=(1, 3, 64, 64))
    except Exception as e:
        print(f"Model not fully implemented yet: {e}")


"""
==============================================================================================
 Epoch  | Train Loss  | Train Acc (%)  |  Val Loss  | Val Acc (%)  | Val F1 (%)  | Time (s)  | Throughput
----------------------------------------------------------------------------------------------
   1    |   1.5546    |     43.97      |   1.3164   |    52.06     |    50.87    |   65.17   |  767.2   s/s
   2    |   1.2201    |     56.39      |   1.1379   |    59.44     |    59.04    |   66.33   |  753.8   s/s
   3    |   1.0763    |     61.83      |   1.0655   |    62.45     |    62.17    |   61.92   |  807.5   s/s
   4    |   0.9697    |     65.68      |   1.0166   |    64.73     |    64.44    |   59.60   |  839.0   s/s
   5    |   0.8888    |     68.78      |   1.0118   |    64.51     |    64.00    |   59.75   |  836.9   s/s
==============================================================================================
Training completed in 313.00s.
Best Val Accuracy: 64.73% | Best Val Macro F1: 64.44%
Model Complexity : 0.446M Parameters | 9.19M FLOPs
Best model weights saved to: checkpoints/lenet_modern
Training curves successfully saved to: plots/loss_lenet_modern.png
"""