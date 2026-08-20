import os
import unittest
from pathlib import Path
from unittest.mock import patch

from industrial_visual_anomaly_detection.service.settings import (
    InferenceServiceSettings,
)


class InferenceServiceSettingsTests(unittest.TestCase):
    def test_legacy_artifact_settings_are_loaded(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "IVAD_MODEL_ARTIFACT": (
                    "outputs/model-artifacts/test-model"
                ),
                "IVAD_MEMORY_CHUNK_SIZE": "2048",
            },
            clear=True,
        ):
            settings = (
                InferenceServiceSettings.from_environment()
            )

        self.assertEqual(
            Path(
                "outputs/model-artifacts/test-model"
            ).resolve(),
            settings.artifact_path,
        )
        self.assertIsNone(settings.model_registry_path)
        self.assertEqual(2048, settings.memory_chunk_size)

    def test_registry_settings_are_loaded(self) -> None:
        with patch.dict(
            os.environ,
            {
                "IVAD_MODEL_REGISTRY": (
                    "runtime-artifacts/models.json"
                ),
                "IVAD_MEMORY_CHUNK_SIZE": "1024",
            },
            clear=True,
        ):
            settings = (
                InferenceServiceSettings.from_environment()
            )

        self.assertIsNone(settings.artifact_path)
        self.assertEqual(
            Path(
                "runtime-artifacts/models.json"
            ).resolve(),
            settings.model_registry_path,
        )
        self.assertEqual(1024, settings.memory_chunk_size)

    def test_missing_model_source_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "Exactly one",
            ):
                InferenceServiceSettings.from_environment()

    def test_multiple_model_sources_are_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "IVAD_MODEL_ARTIFACT": "artifact",
                "IVAD_MODEL_REGISTRY": "models.json",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "must not both",
            ):
                InferenceServiceSettings.from_environment()

    def test_invalid_memory_chunk_size_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "IVAD_MODEL_ARTIFACT": "artifact",
                "IVAD_MEMORY_CHUNK_SIZE": "zero",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "must be an integer",
            ):
                InferenceServiceSettings.from_environment()

    def test_non_positive_memory_chunk_size_is_rejected(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "IVAD_MODEL_ARTIFACT": "artifact",
                "IVAD_MEMORY_CHUNK_SIZE": "0",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "greater than zero",
            ):
                InferenceServiceSettings.from_environment()

    def test_direct_configuration_requires_one_source(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Exactly one",
        ):
            InferenceServiceSettings(
                artifact_path=None,
                memory_chunk_size=4096,
                model_registry_path=None,
            )


if __name__ == "__main__":
    unittest.main()