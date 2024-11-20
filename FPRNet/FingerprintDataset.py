from typing import final, override

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from step1 import PotentialFingerprintPair


@final
class FingerprintDataset(Dataset):  # pyright: ignore[reportMissingTypeArgument]
    data: tuple[PotentialFingerprintPair] | list[PotentialFingerprintPair]

    __slots__ = ("data", "transform")

    def __init__(
        self, dataset: tuple[PotentialFingerprintPair] | list[PotentialFingerprintPair]
    ) -> None:
        self.data = dataset
        self.transform = transforms.Compose(
            [
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, index: int):
        fingerprint_pair = self.data[index]
        reference_path = fingerprint_pair.reference_image
        comparison_path = fingerprint_pair.comparison_image

        reference_image = Image.open(reference_path).convert("L")
        comparison_image = Image.open(comparison_path).convert("L")

        reference_tensor = self.transform(reference_image)
        comparison_tensor = self.transform(comparison_image)

        if fingerprint_pair.match:
            label = torch.tensor(1, dtype=torch.float)
        else:
            label = torch.tensor(0, dtype=torch.float)

        return reference_tensor, comparison_tensor, label
