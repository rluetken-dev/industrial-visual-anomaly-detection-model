from dataclasses import dataclass


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Describe the configuration of an exported anomaly-detection model."""

    schema_version: int
    dataset: str
    category: str
    backbone: str
    input_size: int
    patch_grid_size: tuple[int, int]
    embedding_dimension: int
    aggregation_method: str
    top_fraction: float
    threshold: float
    memory_fraction: float
    sampling_seed: int
    feature_memory_entries: int