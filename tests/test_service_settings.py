import os
import unittest
from pathlib import Path
from unittest.mock import patch

from industrial_visual_anomaly_detection.service.settings import (
    InferenceServiceSettings,
)


class InferenceServiceSettingsTests(unittest.TestCase):
    def test_settings_are_loaded_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "IVAD_MODEL_ARTIFACT": "outputs/model-artifacts/test-model",
                "IVAD_MEMORY_CHUNK_SIZE": "2048",
            },
            clear=True,
        ):
            settings = InferenceServiceSettings.from_environment()

        self.assertEqual(
            Path(
                "outputs/model-artifacts/test-model"
            ).resolve(),
            settings.artifact_path,
        )
        self.assertEqual(2048, settings.memory_chunk_size)

    def test_missing_artifact_path_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "IVAD_MODEL_ARTIFACT",
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


if __name__ == "__main__":
    unittest.main()