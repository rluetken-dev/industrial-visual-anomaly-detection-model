import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from industrial_visual_anomaly_detection.service.model_registry_config import (
    ModelRegistryConfiguration,
    ModelRegistryEntry,
)
from industrial_visual_anomaly_detection.service.runtime import (
    InferenceRuntime,
)
from industrial_visual_anomaly_detection.service.runtime_registry import (
    InferenceRuntimeRegistry,
    UnknownModelError,
)


class InferenceRuntimeRegistryTests(unittest.TestCase):
    def test_default_and_explicit_runtimes_are_selected(
        self,
    ) -> None:
        configuration = self._create_configuration()
        capsule_runtime = self._create_runtime(
            "capsule",
            "capsule",
        )
        cashew_runtime = self._create_runtime(
            "cashew",
            "cashew",
        )

        registry = InferenceRuntimeRegistry(
            configuration=configuration,
            runtimes={
                "capsule": capsule_runtime,
                "cashew": cashew_runtime,
            },
        )

        self.assertIs(
            capsule_runtime,
            registry.get_runtime(),
        )
        self.assertIs(
            cashew_runtime,
            registry.get_runtime("cashew"),
        )

    def test_available_models_expose_registry_and_artifact_data(
        self,
    ) -> None:
        configuration = self._create_configuration()
        registry = InferenceRuntimeRegistry(
            configuration=configuration,
            runtimes={
                "capsule": self._create_runtime(
                    "capsule",
                    "capsule",
                    input_size=320,
                ),
                "cashew": self._create_runtime(
                    "cashew",
                    "cashew",
                    input_size=320,
                ),
            },
        )

        models = registry.available_models

        self.assertEqual(2, len(models))

        self.assertEqual("capsule", models[0].model_id)
        self.assertEqual(
            "MVTec AD - Capsule",
            models[0].display_name,
        )
        self.assertEqual("capsule", models[0].category)
        self.assertEqual(320, models[0].input_size)
        self.assertTrue(models[0].is_default)

        self.assertEqual("cashew", models[1].model_id)
        self.assertEqual(
            "VisA - Cashew",
            models[1].display_name,
        )
        self.assertEqual("cashew", models[1].category)
        self.assertFalse(models[1].is_default)

    def test_unknown_model_is_rejected(self) -> None:
        registry = InferenceRuntimeRegistry(
            configuration=self._create_configuration(),
            runtimes={
                "capsule": self._create_runtime(
                    "capsule",
                    "capsule",
                ),
                "cashew": self._create_runtime(
                    "cashew",
                    "cashew",
                ),
            },
        )

        with self.assertRaisesRegex(
            UnknownModelError,
            "Unknown model ID",
        ):
            registry.get_runtime("missing")

    def test_empty_model_id_is_rejected(self) -> None:
        registry = InferenceRuntimeRegistry(
            configuration=self._create_configuration(),
            runtimes={
                "capsule": self._create_runtime(
                    "capsule",
                    "capsule",
                ),
                "cashew": self._create_runtime(
                    "cashew",
                    "cashew",
                ),
            },
        )

        with self.assertRaisesRegex(
            UnknownModelError,
            "must not be empty",
        ):
            registry.get_runtime(" ")

    def test_runtime_ids_must_match_enabled_models(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "must match enabled",
        ):
            InferenceRuntimeRegistry(
                configuration=self._create_configuration(),
                runtimes={
                    "capsule": self._create_runtime(
                        "capsule",
                        "capsule",
                    ),
                },
            )

    @patch(
        "industrial_visual_anomaly_detection.service."
        "runtime_registry."
        "create_resnet18_patch_embedding_extractor"
    )
    @patch(
        "industrial_visual_anomaly_detection.service."
        "runtime_registry.load_model_artifact"
    )
    def test_enabled_registry_models_are_loaded(
        self,
        load_model_artifact: Mock,
        create_embedding_extractor: Mock,
    ) -> None:
        configuration = self._create_configuration(
            include_disabled=True
        )
        capsule_artifact = SimpleNamespace(
            metadata=SimpleNamespace(
                category="capsule",
                input_size=320,
            )
        )
        cashew_artifact = SimpleNamespace(
            metadata=SimpleNamespace(
                category="cashew",
                input_size=320,
            )
        )
        capsule_extractor = Mock()
        cashew_extractor = Mock()

        load_model_artifact.side_effect = [
            capsule_artifact,
            cashew_artifact,
        ]
        create_embedding_extractor.side_effect = [
            capsule_extractor,
            cashew_extractor,
        ]

        registry = InferenceRuntimeRegistry.load(
            configuration=configuration,
            memory_chunk_size=2048,
        )

        self.assertEqual(
            [
                call(Path("artifacts/capsule")),
                call(Path("artifacts/cashew")),
            ],
            load_model_artifact.call_args_list,
        )
        self.assertEqual(
            2,
            create_embedding_extractor.call_count,
        )

        capsule_runtime = registry.get_runtime("capsule")
        cashew_runtime = registry.get_runtime("cashew")

        self.assertIs(
            capsule_artifact,
            capsule_runtime.artifact,
        )
        self.assertIs(
            capsule_extractor,
            capsule_runtime.embedding_extractor,
        )
        self.assertEqual(
            "capsule",
            capsule_runtime.model_id,
        )
        self.assertEqual(
            2048,
            capsule_runtime.memory_chunk_size,
        )

        self.assertIs(
            cashew_artifact,
            cashew_runtime.artifact,
        )
        self.assertIs(
            cashew_extractor,
            cashew_runtime.embedding_extractor,
        )

    @patch(
        "industrial_visual_anomaly_detection.service."
        "runtime_registry.load_model_artifact"
    )
    def test_invalid_memory_chunk_size_is_rejected(
        self,
        load_model_artifact: Mock,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "greater than zero",
        ):
            InferenceRuntimeRegistry.load(
                configuration=self._create_configuration(),
                memory_chunk_size=0,
            )

        load_model_artifact.assert_not_called()

    @staticmethod
    def _create_configuration(
        include_disabled: bool = False,
    ) -> ModelRegistryConfiguration:
        models = [
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
        ]

        if include_disabled:
            models.append(
                ModelRegistryEntry(
                    model_id="disabled",
                    display_name="Disabled",
                    artifact_path=Path("artifacts/disabled"),
                    enabled=False,
                )
            )

        return ModelRegistryConfiguration(
            default_model_id="capsule",
            models=tuple(models),
        )

    @staticmethod
    def _create_runtime(
        model_id: str,
        category: str,
        input_size: int = 320,
    ) -> InferenceRuntime:
        return InferenceRuntime(
            artifact_path=Path("artifacts") / model_id,
            artifact=SimpleNamespace(
                metadata=SimpleNamespace(
                    category=category,
                    input_size=input_size,
                )
            ),
            embedding_extractor=Mock(),
            memory_chunk_size=4096,
            model_id=model_id,
        )


if __name__ == "__main__":
    unittest.main()