from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSplitManifest:
    """Describe deterministic fitting and validation dataset partitions."""

    schema_version: int
    dataset: str
    category: str
    seed: int
    source_image_count: int
    fitting_image_count: int
    validation_image_count: int
    fitting_images: tuple[Path, ...]
    validation_images: tuple[Path, ...]