import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from industrial_visual_anomaly_detection.service.runtime import (
    InferenceRuntime,
)
from industrial_visual_anomaly_detection.service.settings import (
    InferenceServiceSettings,
)


class InferenceRuntimeTests(unittest.TestCase):
    @patch(
        "industrial_visual_anomaly_detection.service.runtime."
        "create_resnet18_patch_embedding_extractor"
    )
    @patch(
        "industrial_visual_anomaly_detection.service.runtime."
        "load_model_artifact"
    )
    def test_runtime_loads_artifact_and_extractor_once(
        self,
        load_model_artifact: Mock,
        create_embedding_extractor: Mock,
    ) -> None:
        artifact_path = Path("artifacts/capsule").resolve()
        settings = InferenceServiceSettings(
            artifact_path=artifact_path,
            memory_chunk_size=2048,
        )
        artifact = Mock()
        embedding_extractor = Mock()
        load_model_artifact.return_value = artifact
        create_embedding_extractor.return_value = embedding_extractor

        runtime = InferenceRuntime.load(settings)

        load_model_artifact.assert_called_once_with(artifact_path)
        create_embedding_extractor.assert_called_once_with()
        self.assertIs(artifact, runtime.artifact)
        self.assertIs(embedding_extractor, runtime.embedding_extractor)
        self.assertEqual(2048, runtime.memory_chunk_size)

    def test_model_information_is_exposed(self) -> None:
        artifact_path = Path("artifacts/mvtec-ad-capsule-320")
        artifact = SimpleNamespace(
            metadata=SimpleNamespace(category="capsule")
        )
        runtime = InferenceRuntime(
            artifact_path=artifact_path,
            artifact=artifact,
            embedding_extractor=Mock(),
            memory_chunk_size=4096,
        )

        self.assertEqual("mvtec-ad-capsule-320", runtime.model_id)
        self.assertEqual("capsule", runtime.category)

    @patch(
        "industrial_visual_anomaly_detection.service.runtime."
        "predict_image_stream"
    )
    def test_prediction_uses_loaded_runtime_components(
        self,
        predict_image_stream: Mock,
    ) -> None:
        artifact = Mock()
        embedding_extractor = Mock()
        prediction = Mock()
        image_stream = BytesIO(b"image-content")
        predict_image_stream.return_value = prediction

        runtime = InferenceRuntime(
            artifact_path=Path("artifacts/capsule"),
            artifact=artifact,
            embedding_extractor=embedding_extractor,
            memory_chunk_size=1024,
        )

        result = runtime.predict(image_stream)

        self.assertIs(prediction, result)
        predict_image_stream.assert_called_once_with(
            artifact=artifact,
            image_stream=image_stream,
            embedding_extractor=embedding_extractor,
            memory_chunk_size=1024,
        )
    def test_explicit_model_id_overrides_artifact_directory_name(
        self,
    ) -> None:
        runtime = InferenceRuntime(
            artifact_path=Path("artifacts/storage-directory"),
            artifact=SimpleNamespace(
                metadata=SimpleNamespace(category="cashew")
            ),
            embedding_extractor=Mock(),
            memory_chunk_size=4096,
            model_id="visa-cashew-q95-320",
        )

        self.assertEqual(
            "visa-cashew-q95-320",
            runtime.model_id,
        )

if __name__ == "__main__":
    unittest.main()