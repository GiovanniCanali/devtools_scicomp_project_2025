from torch.nn import (
    Module,
    Sequential,
    AdaptiveAvgPool2d,
    Conv2d,
    Linear,
    ReLU,
    MaxPool2d,
    Dropout
)

class AlexNet(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.features = Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),

            Conv2d(64, 192, kernel_size=5, padding=2, groups=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),

            Conv2d(192, 384, kernel_size=3, padding=1, groups=2),
            ReLU(inplace=True),

            Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            ReLU(inplace=True),

            Conv2d(256, 256, kernel_size=3, padding=1, groups=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = AdaptiveAvgPool2d((6, 6))
        self.classifier = Sequential(
            Dropout(),
            Linear(256*6*6, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.avgpool(self.features(x)).flatten(start_dim=1)
        logits = self.classifier(x)
        return logits
