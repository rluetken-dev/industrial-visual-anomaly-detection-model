import unittest
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from industrial_visual_anomaly_detection.service.app import (
    create_app,
)
from industrial_visual_anomaly_detection.service.model_registry_config import (
    ModelRegistryConfiguration,
    ModelRegistryEntry,
)
from industrial_visual_anomaly_detection.service.runtime_registry import (
    InferenceRuntimeRegistry,
)


class ModelCatalogEndpointTests(unittest.TestCase):
    def test_legacy_runtime_is_exposed_as_single_model(
        self,
    ) -> None:
        runtime = Mock()
        runtime.model_id = "mvtec-ad-metal-nut-320"
        runtime.category = "metal_nut"
        runtime.input_size = 320
        app = create_app(runtime=runtime)

        with TestClient(app) as client:
            response = client.get("/api/v1/models")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "defaultModelId": "mvtec-ad-metal-nut-320",
                "models": [
                    {
                        "id": "mvtec-ad-metal-nut-320",
                        "displayName": "Metal Nut",
                        "category": "metal_nut",
                        "inputSize": 320,
                        "isDefault": True,
                    }
                ],
            },
            response.json(),
        )

    def test_registry_models_are_exposed_in_configured_order(
        self,
    ) -> None:
        capsule_runtime = Mock()
        capsule_runtime.model_id = "capsule"
        capsule_runtime.category = "capsule"
        capsule_runtime.input_size = 320

        cashew_runtime = Mock()
        cashew_runtime.model_id = "cashew"
        cashew_runtime.category = "cashew"
        cashew_runtime.input_size = 320

        configuration = ModelRegistryConfiguration(
            default_model_id="capsule",
            models=(
                ModelRegistryEntry(
                    model_id="capsule",
                    display_name="MVTec AD - Capsule",
                    artifact_path=Path("artifacts/capsule"),
                    enabled=True,
                ),
                ModelRegistryEntry(
                    model_id="cashew",
                    display_name="VisA - Cashew",
                    artifact_path=Path("artifacts/cashew"),
                    enabled=True,
                ),
                ModelRegistryEntry(
                    model_id="disabled",
                    display_name="Disabled",
                    artifact_path=Path("artifacts/disabled"),
                    enabled=False,
                ),
            ),
        )
        registry = InferenceRuntimeRegistry(
            configuration=configuration,
            runtimes={
                "capsule": capsule_runtime,
                "cashew": cashew_runtime,
            },
        )
        app = create_app(runtime=registry)

        with TestClient(app) as client:
            response = client.get("/api/v1/models")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "defaultModelId": "capsule",
                "models": [
                    {
                        "id": "capsule",
                        "displayName": "MVTec AD - Capsule",
                        "category": "capsule",
                        "inputSize": 320,
                        "isDefault": True,
                    },
                    {
                        "id": "cashew",
                        "displayName": "VisA - Cashew",
                        "category": "cashew",
                        "inputSize": 320,
                        "isDefault": False,
                    },
                ],
            },
            response.json(),
        )


if __name__ == "__main__":
    unittest.main()