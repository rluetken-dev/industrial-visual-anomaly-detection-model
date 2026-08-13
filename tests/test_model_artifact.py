import tempfile
import unittest
from pathlib import Path

import torch

from industrial_visual_anomaly_detection.artifacts import (
    ModelArtifact,
    ModelArtifactMetadata,
    load_model_artifact,
    save_model_artifact,
)


class ModelArtifactTests(unittest.TestCase):
    def create_metadata(
        self,
        feature_memory_entries: int = 2,
        embedding_dimension: int = 3,
    ) -> ModelArtifactMetadata:
        return ModelArtifactMetadata(
            schema_version=1,
            dataset="test-dataset",
            category="test-category",
            backbone="resnet18",
            input_size=320,
            patch_grid_size=(40, 40),
            embedding_dimension=embedding_dimension,
            aggregation_method="top_fraction_mean",
            top_fraction=0.01,
            threshold=2.5,
            memory_fraction=1.0,
            sampling_seed=42,
            feature_memory_entries=feature_memory_entries,
        )

    def test_artifact_round_trip_preserves_data(self) -> None:
        feature_memory = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        artifact = ModelArtifact(
            metadata=self.create_metadata(),
            feature_memory=feature_memory,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_directory = save_model_artifact(
                artifact,
                Path(temporary_directory) / "model",
            )
            loaded_artifact = load_model_artifact(
                artifact_directory
            )

        self.assertEqual(
            artifact.metadata,
            loaded_artifact.metadata,
        )
        self.assertTrue(
            torch.equal(
                artifact.feature_memory,
                loaded_artifact.feature_memory,
            )
        )

    def test_entry_count_mismatch_is_rejected(self) -> None:
        artifact = ModelArtifact(
            metadata=self.create_metadata(
                feature_memory_entries=3,
            ),
            feature_memory=torch.zeros(2, 3),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError,
                "entry count",
            ):
                save_model_artifact(
                    artifact,
                    Path(temporary_directory) / "model",
                )

    def test_embedding_dimension_mismatch_is_rejected(self) -> None:
        artifact = ModelArtifact(
            metadata=self.create_metadata(
                embedding_dimension=4,
            ),
            feature_memory=torch.zeros(2, 3),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError,
                "dimension",
            ):
                save_model_artifact(
                    artifact,
                    Path(temporary_directory) / "model",
                )

    def test_missing_artifact_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_directory = (
                Path(temporary_directory) / "missing"
            )

            with self.assertRaises(NotADirectoryError):
                load_model_artifact(missing_directory)


if __name__ == "__main__":
    unittest.main()