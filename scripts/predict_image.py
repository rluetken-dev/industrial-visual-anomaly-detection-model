import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from industrial_visual_anomaly_detection.artifacts import (
    load_model_artifact,
)
from industrial_visual_anomaly_detection.inference import predict_image
from industrial_visual_anomaly_detection.models import (
    create_resnet18_patch_embedding_extractor,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Predict whether one image contains a visual anomaly "
            "using an exported model artifact."
        )
    )
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help="Directory containing metadata.json and feature_memory.pt.",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Image file to evaluate.",
    )
    parser.add_argument(
        "--memory-chunk-size",
        type=int,
        default=4096,
        help="Number of feature-memory entries processed per distance chunk.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.memory_chunk_size <= 0:
        raise ValueError("Memory chunk size must be greater than zero.")

    load_start = time.perf_counter()
    artifact = load_model_artifact(arguments.artifact)
    artifact_load_seconds = time.perf_counter() - load_start

    extractor_start = time.perf_counter()
    embedding_extractor = create_resnet18_patch_embedding_extractor()
    extractor_creation_seconds = time.perf_counter() - extractor_start

    prediction_start = time.perf_counter()
    prediction = predict_image(
        artifact,
        arguments.image,
        embedding_extractor,
        memory_chunk_size=arguments.memory_chunk_size,
    )
    prediction_seconds = time.perf_counter() - prediction_start

    metadata = artifact.metadata
    decision = "anomalous" if prediction.is_anomalous else "normal"

    print(f"Image: {prediction.image_path}")
    print(f"Dataset: {metadata.dataset}")
    print(f"Category: {metadata.category}")
    print(f"Backbone: {metadata.backbone}")
    print(f"Input size: {metadata.input_size} x {metadata.input_size}")
    print(f"Patch grid size: {metadata.patch_grid_size}")
    print(f"Aggregation method: {metadata.aggregation_method}")
    print(f"Top fraction: {metadata.top_fraction:.4f}")
    print(f"Feature memory shape: {tuple(artifact.feature_memory.shape)}")
    print(f"Anomaly score: {prediction.anomaly_score:.6f}")
    print(f"Threshold: {prediction.threshold:.6f}")
    print(f"Decision: {decision}")
    print(f"Artifact load time: {artifact_load_seconds:.2f} seconds")
    print(
        "Feature extractor creation time: "
        f"{extractor_creation_seconds:.2f} seconds"
    )
    print(f"Prediction time: {prediction_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
