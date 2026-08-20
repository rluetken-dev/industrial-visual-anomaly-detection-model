import json
import tempfile
import unittest
from pathlib import Path

from industrial_visual_anomaly_detection.service.model_registry_config import (
    load_model_registry_configuration,
)


class ModelRegistryConfigurationTests(unittest.TestCase):
    def test_valid_registry_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "capsule").mkdir()
            (root / "cashew").mkdir()

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "capsule-320",
                    "models": [
                        {
                            "id": "capsule-320",
                            "displayName": "Capsule",
                            "artifactDirectory": "capsule",
                            "enabled": True,
                        },
                        {
                            "id": "cashew-q95-320",
                            "displayName": "VisA Cashew",
                            "artifactDirectory": "cashew",
                            "enabled": True,
                        },
                    ],
                },
            )

            configuration = load_model_registry_configuration(
                registry_path
            )

        self.assertEqual(
            "capsule-320",
            configuration.default_model_id,
        )
        self.assertEqual(2, len(configuration.models))
        self.assertEqual(2, len(configuration.enabled_models))
        self.assertEqual(
            "VisA Cashew",
            configuration.models[1].display_name,
        )
        self.assertEqual(
            (root / "cashew").resolve(),
            configuration.models[1].artifact_path,
        )

    def test_disabled_missing_artifact_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "capsule").mkdir()

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "capsule",
                    "models": [
                        {
                            "id": "capsule",
                            "displayName": "Capsule",
                            "artifactDirectory": "capsule",
                            "enabled": True,
                        },
                        {
                            "id": "future-model",
                            "displayName": "Future model",
                            "artifactDirectory": "missing",
                            "enabled": False,
                        },
                    ],
                },
            )

            configuration = load_model_registry_configuration(
                registry_path
            )

        self.assertEqual(2, len(configuration.models))
        self.assertEqual(1, len(configuration.enabled_models))

    def test_missing_registry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry_path = (
                Path(temporary_directory) / "missing.json"
            )

            with self.assertRaises(FileNotFoundError):
                load_model_registry_configuration(registry_path)

    def test_non_object_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry_path = self._write_registry(
                root,
                [],
            )

            with self.assertRaisesRegex(
                ValueError,
                "root must be a JSON object",
            ):
                load_model_registry_configuration(registry_path)

    def test_unsupported_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 2,
                    "defaultModelId": "model",
                    "models": [],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "schemaVersion must be 1",
            ):
                load_model_registry_configuration(registry_path)

    def test_empty_model_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "model",
                    "models": [],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "non-empty array",
            ):
                load_model_registry_configuration(registry_path)

    def test_duplicate_model_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "first").mkdir()
            (root / "second").mkdir()

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "duplicate",
                    "models": [
                        {
                            "id": "duplicate",
                            "displayName": "First",
                            "artifactDirectory": "first",
                            "enabled": True,
                        },
                        {
                            "id": "duplicate",
                            "displayName": "Second",
                            "artifactDirectory": "second",
                            "enabled": True,
                        },
                    ],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "model IDs must be unique",
            ):
                load_model_registry_configuration(registry_path)

    def test_duplicate_artifact_directories_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "shared").mkdir()

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "first",
                    "models": [
                        {
                            "id": "first",
                            "displayName": "First",
                            "artifactDirectory": "shared",
                            "enabled": True,
                        },
                        {
                            "id": "second",
                            "displayName": "Second",
                            "artifactDirectory": "shared",
                            "enabled": True,
                        },
                    ],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "artifact directories must be unique",
            ):
                load_model_registry_configuration(registry_path)

    def test_registry_without_enabled_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "disabled",
                    "models": [
                        {
                            "id": "disabled",
                            "displayName": "Disabled",
                            "artifactDirectory": "missing",
                            "enabled": False,
                        },
                    ],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "at least one enabled model",
            ):
                load_model_registry_configuration(registry_path)

    def test_disabled_default_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "enabled").mkdir()

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "disabled",
                    "models": [
                        {
                            "id": "enabled",
                            "displayName": "Enabled",
                            "artifactDirectory": "enabled",
                            "enabled": True,
                        },
                        {
                            "id": "disabled",
                            "displayName": "Disabled",
                            "artifactDirectory": "disabled",
                            "enabled": False,
                        },
                    ],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "must reference an enabled model",
            ):
                load_model_registry_configuration(registry_path)

    def test_missing_enabled_artifact_directory_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "missing",
                    "models": [
                        {
                            "id": "missing",
                            "displayName": "Missing",
                            "artifactDirectory": "missing",
                            "enabled": True,
                        },
                    ],
                },
            )

            with self.assertRaises(NotADirectoryError):
                load_model_registry_configuration(registry_path)

    def test_absolute_artifact_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            absolute_artifact_path = (root / "artifact").resolve()

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "model",
                    "models": [
                        {
                            "id": "model",
                            "displayName": "Model",
                            "artifactDirectory": str(
                                absolute_artifact_path
                            ),
                            "enabled": True,
                        },
                    ],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "must be a relative path",
            ):
                load_model_registry_configuration(registry_path)

    def test_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "outside",
                    "models": [
                        {
                            "id": "outside",
                            "displayName": "Outside",
                            "artifactDirectory": "../outside",
                            "enabled": True,
                        },
                    ],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "must stay inside",
            ):
                load_model_registry_configuration(registry_path)

    def test_invalid_model_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "Invalid Model",
                    "models": [
                        {
                            "id": "Invalid Model",
                            "displayName": "Invalid",
                            "artifactDirectory": "model",
                            "enabled": True,
                        },
                    ],
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "lowercase letters",
            ):
                load_model_registry_configuration(registry_path)

    def test_unexpected_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            registry_path = self._write_registry(
                root,
                {
                    "schemaVersion": 1,
                    "defaultModelId": "model",
                    "models": [],
                    "unexpected": True,
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "Unexpected",
            ):
                load_model_registry_configuration(registry_path)

    @staticmethod
    def _write_registry(
        root: Path,
        data: object,
    ) -> Path:
        registry_path = root / "models.json"
        registry_path.write_text(
            json.dumps(data),
            encoding="utf-8",
        )
        return registry_path

    def test_registry_with_utf8_bom_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "model").mkdir()

            registry_path = root / "models.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "defaultModelId": "test-model",
                        "models": [
                            {
                                "id": "test-model",
                                "displayName": "Test Model",
                                "artifactDirectory": "model",
                                "enabled": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8-sig",
            )

            configuration = load_model_registry_configuration(
                registry_path
            )

        self.assertEqual(
            "test-model",
            configuration.default_model_id,
        )
        self.assertEqual(
            "test-model",
            configuration.enabled_models[0].model_id,
        )


if __name__ == "__main__":
    unittest.main()