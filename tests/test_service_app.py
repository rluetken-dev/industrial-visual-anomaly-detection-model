import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from industrial_visual_anomaly_detection.service.app import (
    create_app,
)
from industrial_visual_anomaly_detection.service.settings import (
    InferenceServiceSettings,
)


class InferenceServiceAppTests(unittest.TestCase):
    def test_liveness_endpoint_returns_healthy_status(
        self,
    ) -> None:
        runtime = Mock()
        app = create_app(runtime=runtime)

        with TestClient(app) as client:
            response = client.get("/health/live")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {"status": "healthy"},
            response.json(),
        )

    def test_injected_runtime_is_stored_in_application_state(
        self,
    ) -> None:
        runtime = Mock()
        app = create_app(runtime=runtime)

        with TestClient(app):
            stored_runtime = app.state.inference_runtime

        self.assertIs(runtime, stored_runtime)

    @patch(
        "industrial_visual_anomaly_detection.service.app."
        "InferenceRuntime.load"
    )
    @patch(
        "industrial_visual_anomaly_detection.service.app."
        "InferenceServiceSettings.from_environment"
    )
    def test_legacy_runtime_is_loaded_during_startup(
        self,
        from_environment: Mock,
        load_runtime: Mock,
    ) -> None:
        settings = InferenceServiceSettings(
            artifact_path=Path("artifact"),
            memory_chunk_size=4096,
        )
        runtime = Mock()
        from_environment.return_value = settings
        load_runtime.return_value = runtime
        app = create_app()

        with TestClient(app):
            stored_runtime = app.state.inference_runtime

        from_environment.assert_called_once_with()
        load_runtime.assert_called_once_with(settings)
        self.assertIs(runtime, stored_runtime)

    @patch(
        "industrial_visual_anomaly_detection.service.app."
        "InferenceRuntimeRegistry.load"
    )
    @patch(
        "industrial_visual_anomaly_detection.service.app."
        "load_model_registry_configuration"
    )
    @patch(
        "industrial_visual_anomaly_detection.service.app."
        "InferenceServiceSettings.from_environment"
    )
    def test_registry_runtime_is_loaded_during_startup(
        self,
        from_environment: Mock,
        load_configuration: Mock,
        load_registry: Mock,
    ) -> None:
        registry_path = Path("models.json").resolve()
        settings = InferenceServiceSettings(
            artifact_path=None,
            memory_chunk_size=2048,
            model_registry_path=registry_path,
        )
        configuration = Mock()
        registry = Mock()
        from_environment.return_value = settings
        load_configuration.return_value = configuration
        load_registry.return_value = registry
        app = create_app()

        with TestClient(app):
            stored_runtime = app.state.inference_runtime

        from_environment.assert_called_once_with()
        load_configuration.assert_called_once_with(
            registry_path
        )
        load_registry.assert_called_once_with(
            configuration=configuration,
            memory_chunk_size=2048,
        )
        self.assertIs(registry, stored_runtime)


if __name__ == "__main__":
    unittest.main()