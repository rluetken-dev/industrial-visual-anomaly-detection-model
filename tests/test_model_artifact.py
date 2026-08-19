import json
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
        schema_version: int = 1,
        threshold_method: str = "maximum_normal",
        threshold_quantile: float = 1.0,
    ) -> ModelArtifactMetadata:
        return ModelArtifactMetadata(
            schema_version=schema_version,
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
            threshold_method=threshold_method,
            threshold_quantile=threshold_quantile,
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

    def test_schema_two_threshold_metadata_is_preserved(
        self,
    ) -> None:
        artifact = ModelArtifact(
            metadata=self.create_metadata(
                schema_version=2,
                threshold_method="normal_score_quantile",
                threshold_quantile=0.95,
            ),
            feature_memory=torch.zeros(2, 3),
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
            2,
            loaded_artifact.metadata.schema_version,
        )
        self.assertEqual(
            "normal_score_quantile",
            loaded_artifact.metadata.threshold_method,
        )
        self.assertAlmostEqual(
            0.95,
            loaded_artifact.metadata.threshold_quantile,
        )

    def test_schema_one_without_threshold_metadata_uses_defaults(
        self,
    ) -> None:
        artifact = ModelArtifact(
            metadata=self.create_metadata(),
            feature_memory=torch.zeros(2, 3),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_directory = save_model_artifact(
                artifact,
                Path(temporary_directory) / "model",
            )
            metadata_path = (
                artifact_directory / "metadata.json"
            )

            metadata_data = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
            metadata_data.pop("threshold_method")
            metadata_data.pop("threshold_quantile")
            metadata_path.write_text(
                json.dumps(
                    metadata_data,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            loaded_artifact = load_model_artifact(
                artifact_directory
            )

        self.assertEqual(
            1,
            loaded_artifact.metadata.schema_version,
        )
        self.assertEqual(
            "maximum_normal",
            loaded_artifact.metadata.threshold_method,
        )
        self.assertAlmostEqual(
            1.0,
            loaded_artifact.metadata.threshold_quantile,
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