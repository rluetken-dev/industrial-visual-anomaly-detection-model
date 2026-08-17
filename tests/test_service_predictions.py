from base64 import b64decode
from io import BytesIO
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image, UnidentifiedImageError
import torch

from industrial_visual_anomaly_detection.service.app import create_app


class PredictionEndpointTests(unittest.TestCase):
    def test_uploaded_image_returns_prediction_response(self) -> None:
        runtime = Mock()
        runtime.model_id = "mvtec-ad-capsule-320"
        runtime.category = "capsule"
        runtime.input_size = 8
        runtime.predict.return_value = SimpleNamespace(
            anomaly_score=4.992109,
            threshold=2.501822,
            is_anomalous=True,
            patch_scores=torch.tensor(
                [
                    [0.0, 1.0],
                    [2.0, 4.0],
                ]
            ),
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

        response_data = response.json()
        heatmap_data = response_data.pop("heatmap")

        self.assertEqual(
            {
                "modelId": "mvtec-ad-capsule-320",
                "category": "capsule",
                "score": 4.992109,
                "threshold": 2.501822,
                "isAnomalous": True,
            },
            response_data,
        )
        self.assertEqual("image/png", heatmap_data["contentType"])
        self.assertEqual(8, heatmap_data["width"])
        self.assertEqual(8, heatmap_data["height"])

        with Image.open(
            BytesIO(b64decode(heatmap_data["dataBase64"]))
        ) as heatmap:
            self.assertEqual("PNG", heatmap.format)
            self.assertEqual("RGB", heatmap.mode)
            self.assertEqual((8, 8), heatmap.size)

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