from typing import Any, final, override

import torch
from torch import nn

# pyright: basic


@final
class FeatureExtractor(nn.Module):
    conv32: nn.Conv2d
    conv32to64: nn.Conv2d
    conv64: nn.Conv2d
    conv64to128: nn.Conv2d
    conv128: nn.Conv2d

    bn32: nn.BatchNorm2d
    bn64: nn.BatchNorm2d
    bn128: nn.BatchNorm2d

    pool: nn.MaxPool2d
    relu: nn.ReLU

    __slots__ = (
        "conv32",
        "conv32to64",
        "conv64",
        "conv64to128",
        "conv128",
        "bn32",
        "bn64",
        "bn128",
        "pool",
        "relu",
    )

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
    def forward(self, x: Any) -> Any:
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


@final
class FPRNet(nn.Module):
    feature_extractor1: FeatureExtractor
    feature_extractor2: FeatureExtractor
    conv128to256: nn.Conv2d
    bn256: nn.BatchNorm2d
    bn1024: nn.BatchNorm1d
    dense1: nn.Linear
    dense1024: nn.Linear
    flatten: nn.Flatten
    dropout: nn.Dropout
    relu: nn.ReLU
    sigmoid: nn.Sigmoid

    __slots__ = (
        "feature_extractor1",
        "feature_extractor2",
        "conv128to256",
        "bn256",
        "bn1024",
        "dense1",
        "dense1024",
        "flatten",
        "dropout",
        "relu",
        "sigmoid",
    )

    def __init__(self) -> None:
        super().__init__()

        self.feature_extractor1 = FeatureExtractor()
        self.feature_extractor2 = FeatureExtractor()

        self.conv128to256 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn256 = nn.BatchNorm2d(256)
        self.dense1024 = nn.Linear(256 * 1024, 1024)
        self.bn1024 = nn.BatchNorm1d(1024)
        self.dense1 = nn.Linear(1024, 1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    @override
    def forward(self, image1: Any, image2: Any) -> Any:
        feature1 = self.feature_extractor1(image1)
        feature2 = self.feature_extractor2(image2)

        x = torch.abs(feature1 - feature2)
        x = self.conv128to256(x)
        x = self.bn256(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dense1024(x)
        x = self.bn1024(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.sigmoid(x)
        return x
