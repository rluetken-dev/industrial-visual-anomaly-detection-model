import time
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from industrial_visual_anomaly_detection.artifacts import (
    ModelArtifact,
    ModelArtifactMetadata,
    save_model_artifact,
)
from industrial_visual_anomaly_detection.datasets import ImagePathDataset
from industrial_visual_anomaly_detection.evaluation import (
    select_normal_score_quantile_threshold,
)
from industrial_visual_anomaly_detection.models import (
    aggregate_top_patch_scores,
    build_feature_memory,
    compute_patch_scores_for_batches,
    create_resnet18_patch_embedding_extractor,
    sample_feature_memory,
)
from industrial_visual_anomaly_detection.preprocessing import (
    create_image_preprocessing,
)


@dataclass(frozen=True)
class ModelTrainingConfiguration:
    """Configure feature-memory construction and artifact export."""

    batch_size: int = 8
    memory_chunk_size: int = 4096
    input_size: int = 224
    top_fraction: float = 0.01
    threshold_quantile: float = 1.0
    memory_fraction: float = 1.0
    sampling_seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if self.memory_chunk_size <= 0:
            raise ValueError(
                "Memory chunk size must be greater than zero."
            )

        if self.input_size <= 0:
            raise ValueError("Input size must be greater than zero.")

        if self.input_size % 32 != 0:
            raise ValueError("Input size must be divisible by 32.")

        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError(
                "Top fraction must be greater than zero "
                "and at most one."
            )

        if not 0.0 < self.threshold_quantile <= 1.0:
            raise ValueError(
                "Threshold quantile must be greater than zero "
                "and at most one."
            )

        if not 0.0 < self.memory_fraction <= 1.0:
            raise ValueError(
                "Memory fraction must be greater than zero "
                "and at most one."
            )


@dataclass(frozen=True)
class ModelTrainingResult:
    """Describe a completed model-artifact export."""

    artifact_directory: Path
    fitting_image_count: int
    validation_image_count: int
    complete_memory_shape: tuple[int, ...]
    exported_memory_shape: tuple[int, ...]
    feature_memory_size_mib: float
    threshold: float
    memory_seconds: float
    validation_seconds: float
    export_seconds: float


def train_model_artifact(
    fitting_paths: tuple[Path, ...],
    validation_paths: tuple[Path, ...],
    dataset: str,
    category: str,
    output_directory: Path,
    configuration: ModelTrainingConfiguration,
) -> ModelTrainingResult:
    """Train and export one anomaly-detection model artifact."""

    if not fitting_paths:
        raise ValueError("Fitting images are required.")

    if not validation_paths:
        raise ValueError("Validation images are required.")

    if set(fitting_paths) & set(validation_paths):
        raise ValueError(
            "Fitting and validation image paths must not overlap."
        )

    if not dataset.strip():
        raise ValueError("Dataset name is required.")

    if not category.strip():
        raise ValueError("Category name is required.")

    preprocessing = create_image_preprocessing(
        input_size=(
            configuration.input_size,
            configuration.input_size,
        )
    )

    fitting_dataset = ImagePathDataset(
        fitting_paths,
        preprocessing,
    )
    validation_dataset = ImagePathDataset(
        validation_paths,
        preprocessing,
    )

    fitting_loader = DataLoader(
        fitting_dataset,
        batch_size=configuration.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=configuration.batch_size,
        shuffle=False,
        num_workers=0,
    )

    embedding_extractor = (
        create_resnet18_patch_embedding_extractor()
    )

    memory_start = time.perf_counter()

    complete_feature_memory = build_feature_memory(
        fitting_loader,
        embedding_extractor,
    )
    complete_memory_shape = tuple(complete_feature_memory.shape)

    feature_memory = sample_feature_memory(
        complete_feature_memory,
        fraction=configuration.memory_fraction,
        seed=configuration.sampling_seed,
    )
    feature_memory = feature_memory.contiguous()
    exported_memory_shape = tuple(feature_memory.shape)

    del complete_feature_memory

    memory_seconds = time.perf_counter() - memory_start

    patch_grid_side = configuration.input_size // 8
    patch_grid_size = (patch_grid_side, patch_grid_side)

    validation_start = time.perf_counter()

    validation_patch_scores, validation_scored_paths = (
        compute_patch_scores_for_batches(
            validation_loader,
            embedding_extractor,
            feature_memory,
            memory_chunk_size=configuration.memory_chunk_size,
            patch_grid_size=patch_grid_size,
        )
    )

    validation_seconds = (
        time.perf_counter() - validation_start
    )

    expected_validation_paths = tuple(
        str(path) for path in validation_paths
    )

    if validation_scored_paths != expected_validation_paths:
        raise RuntimeError(
            "Validation score paths do not match the input order."
        )

    validation_scores = aggregate_top_patch_scores(
        validation_patch_scores,
        top_fraction=configuration.top_fraction,
    )
    threshold = select_normal_score_quantile_threshold(
        validation_scores,
        quantile=configuration.threshold_quantile,
    )

    metadata = ModelArtifactMetadata(
        schema_version=2,
        dataset=dataset,
        category=category,
        backbone="resnet18",
        input_size=configuration.input_size,
        patch_grid_size=patch_grid_size,
        embedding_dimension=feature_memory.shape[1],
        aggregation_method="top_fraction_mean",
        top_fraction=configuration.top_fraction,
        threshold=threshold,
        memory_fraction=configuration.memory_fraction,
        sampling_seed=configuration.sampling_seed,
        feature_memory_entries=feature_memory.shape[0],
        threshold_method="normal_score_quantile",
        threshold_quantile=(
            configuration.threshold_quantile
        ),
    )
    artifact = ModelArtifact(
        metadata=metadata,
        feature_memory=feature_memory,
    )

    export_start = time.perf_counter()

    artifact_directory = save_model_artifact(
        artifact,
        output_directory,
    )

    export_seconds = time.perf_counter() - export_start

    feature_memory_size_mib = (
        feature_memory.numel()
        * feature_memory.element_size()
        / (1024 * 1024)
    )

    return ModelTrainingResult(
        artifact_directory=artifact_directory,
        fitting_image_count=len(fitting_dataset),
        validation_image_count=len(validation_dataset),
        complete_memory_shape=complete_memory_shape,
        exported_memory_shape=exported_memory_shape,
        feature_memory_size_mib=feature_memory_size_mib,
        threshold=threshold,
        memory_seconds=memory_seconds,
        validation_seconds=validation_seconds,
        export_seconds=export_seconds,
    )