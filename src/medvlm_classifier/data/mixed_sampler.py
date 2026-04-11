from __future__ import annotations

import random
from typing import Any

from torch.utils.data import Dataset


class MixedBatchDataset(Dataset):
    """Virtual mixed dataset that samples from two datasets with a fixed ratio.

    Example: cls_to_inst_ratio=(2,1) means 2/3 samples from classification data.
    """

    def __init__(
        self,
        classification_dataset: Dataset,
        instruction_dataset: Dataset,
        cls_to_inst_ratio: tuple[int, int] = (2, 1),
        virtual_size: int | None = None,
        seed: int = 42,
    ):
        if cls_to_inst_ratio[0] <= 0 or cls_to_inst_ratio[1] <= 0:
            raise ValueError("Sampling ratio values must be positive integers.")
        self.classification_dataset = classification_dataset
        self.instruction_dataset = instruction_dataset
        self.cls_to_inst_ratio = cls_to_inst_ratio
        self.random = random.Random(seed)

        base_len = max(len(classification_dataset), len(instruction_dataset))
        self._size = virtual_size if virtual_size is not None else base_len * sum(cls_to_inst_ratio)

        cls_weight = cls_to_inst_ratio[0]
        inst_weight = cls_to_inst_ratio[1]
        total = cls_weight + inst_weight
        self.cls_prob = cls_weight / total

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        _ = idx
        if self.random.random() < self.cls_prob:
            sampled_idx = self.random.randrange(len(self.classification_dataset))
            return self.classification_dataset[sampled_idx]
        sampled_idx = self.random.randrange(len(self.instruction_dataset))
        return self.instruction_dataset[sampled_idx]
