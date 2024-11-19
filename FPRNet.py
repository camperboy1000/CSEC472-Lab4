from typing import final, override

from torch import nn

# pyright: basic


@final
class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv32 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv32to64 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv64 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv64to128 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv128 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    @override
    def forward(self, x):
        # Block 1
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv32to64(x)
        x = self.bn64(x)
        x = self.relu(x)
        x = self.conv64(x)
        x = self.bn64(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv64(x)
        x = self.bn64(x)
        x = self.relu(x)
        x = self.conv64(x)
        x = self.bn64(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 4
        x = self.conv64to128(x)
        x = self.bn128(x)
        x = self.relu(x)
        x = self.conv128(x)
        x = self.bn128(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
