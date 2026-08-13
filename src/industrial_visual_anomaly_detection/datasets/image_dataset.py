from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImagePathDataset(Dataset[tuple[torch.Tensor, str]]):
    """Load images from fixed paths and apply deterministic preprocessing."""

    def __init__(
        self,
        image_paths: tuple[Path, ...],
        preprocessing: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        if not image_paths:
            raise ValueError("Image dataset must contain at least one path.")

        self.image_paths = image_paths
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[index]

        with Image.open(image_path) as source_image:
            rgb_image = source_image.convert("RGB")
            tensor = self.preprocessing(rgb_image)

        return tensor, str(image_path)