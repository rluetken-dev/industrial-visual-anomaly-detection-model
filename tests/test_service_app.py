import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from industrial_visual_anomaly_detection.service.app import create_app


class InferenceServiceAppTests(unittest.TestCase):
    def test_liveness_endpoint_returns_healthy_status(self) -> None:
        runtime = Mock()
        app = create_app(runtime=runtime)

        with TestClient(app) as client:
            response = client.get("/health/live")

        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "healthy"}, response.json())

    def test_injected_runtime_is_stored_in_application_state(self) -> None:
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
    def test_runtime_is_loaded_during_application_startup(
        self,
        from_environment: Mock,
        load_runtime: Mock,
    ) -> None:
        settings = Mock()
        runtime = Mock()
        from_environment.return_value = settings
        load_runtime.return_value = runtime
        app = create_app()

        with TestClient(app):
            stored_runtime = app.state.inference_runtime

        from_environment.assert_called_once_with()
        load_runtime.assert_called_once_with(settings)
        self.assertIs(runtime, stored_runtime)


if __name__ == "__main__":
    unittest.main()