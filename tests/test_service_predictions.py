import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from industrial_visual_anomaly_detection.service.app import create_app

from PIL import UnidentifiedImageError


class PredictionEndpointTests(unittest.TestCase):
    def test_uploaded_image_returns_prediction_response(self) -> None:
        runtime = Mock()
        runtime.model_id = "mvtec-ad-capsule-320"
        runtime.category = "capsule"
        runtime.predict.return_value = SimpleNamespace(
            anomaly_score=4.992109,
            threshold=2.501822,
            is_anomalous=True,
        )
        app = create_app(runtime=runtime)

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/predictions",
                files={
                    "image": (
                        "capsule.png",
                        b"image-content",
                        "image/png",
                    )
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "modelId": "mvtec-ad-capsule-320",
                "category": "capsule",
                "score": 4.992109,
                "threshold": 2.501822,
                "isAnomalous": True,
            },
            response.json(),
        )
        runtime.predict.assert_called_once()

    def test_missing_image_returns_validation_error(self) -> None:
        runtime = Mock()
        app = create_app(runtime=runtime)

        with TestClient(app) as client:
            response = client.post("/api/v1/predictions")

        self.assertEqual(422, response.status_code)
        runtime.predict.assert_not_called()

    def test_unreadable_image_returns_bad_request(self) -> None:
        runtime = Mock()
        runtime.predict.side_effect = UnidentifiedImageError(
            "cannot identify image file"
        )
        app = create_app(runtime=runtime)

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/predictions",
                files={
                    "image": (
                        "invalid.png",
                        b"\x89PNG\r\n\x1a\n",
                        "image/png",
                    )
                },
            )

        self.assertEqual(400, response.status_code)
        self.assertEqual(
            {"detail": "Uploaded file is not a readable image."},
            response.json(),
        )
        runtime.predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()