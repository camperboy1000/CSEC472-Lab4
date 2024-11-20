from typing import final, override

import torch
from PIL import Image
from torch.utils.data import Dataset

from step1 import PotentialFingerprintPair


@final
class FingerprintDataset(Dataset):  # pyright: ignore[reportMissingTypeArgument]
    data: tuple[PotentialFingerprintPair] | list[PotentialFingerprintPair]

    __slots__ = "data"

    def __init__(
        self, dataset: tuple[PotentialFingerprintPair] | list[PotentialFingerprintPair]
    ) -> None:
        self.data = dataset

    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, index: int):
        fingerprint_pair = self.data[index]
        reference_path = fingerprint_pair.reference_image
        comparision_path = fingerprint_pair.comparision_image

        reference_image = Image.open(reference_path).convert("L")
        comparision_image = Image.open(comparision_path).convert("L")

        if fingerprint_pair.match:
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)

        return reference_image, comparision_image, label
