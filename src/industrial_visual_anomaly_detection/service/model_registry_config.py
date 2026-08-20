import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Describe one configured model artifact."""

    model_id: str
    display_name: str
    artifact_path: Path
    enabled: bool


@dataclass(frozen=True)
class ModelRegistryConfiguration:
    """Contain a validated multi-model registry configuration."""

    default_model_id: str
    models: tuple[ModelRegistryEntry, ...]

    @property
    def enabled_models(self) -> tuple[ModelRegistryEntry, ...]:
        """Return enabled registry entries."""

        return tuple(
            model for model in self.models if model.enabled
        )


def load_model_registry_configuration(
    registry_path: Path,
) -> ModelRegistryConfiguration:
    """Load and validate a model-registry JSON file."""

    resolved_registry_path = registry_path.expanduser().resolve()

    if not resolved_registry_path.is_file():
        raise FileNotFoundError(
            f"Model registry does not exist: "
            f"{resolved_registry_path}"
        )

    with resolved_registry_path.open(
        encoding="utf-8-sig",
    ) as registry_file:
        data = json.load(registry_file)

    if not isinstance(data, dict):
        raise ValueError(
            "Model registry root must be a JSON object."
        )

    _require_exact_keys(
        data,
        {"schemaVersion", "defaultModelId", "models"},
        "Model registry",
    )

    schema_version = data["schemaVersion"]

    if type(schema_version) is not int or schema_version != 1:
        raise ValueError(
            "Model registry schemaVersion must be 1."
        )

    default_model_id = _validate_model_id(
        data["defaultModelId"],
        "defaultModelId",
    )

    model_values = data["models"]

    if not isinstance(model_values, list) or not model_values:
        raise ValueError(
            "Model registry models must be a non-empty array."
        )

    registry_root = resolved_registry_path.parent
    models = tuple(
        _parse_model_entry(
            value,
            index,
            registry_root,
        )
        for index, value in enumerate(model_values)
    )

    model_ids = [model.model_id for model in models]

    if len(set(model_ids)) != len(model_ids):
        raise ValueError(
            "Model registry model IDs must be unique."
        )

    artifact_paths = [
        model.artifact_path for model in models
    ]

    if len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError(
            "Model registry artifact directories must be unique."
        )

    enabled_models = tuple(
        model for model in models if model.enabled
    )

    if not enabled_models:
        raise ValueError(
            "Model registry must contain at least one enabled model."
        )

    enabled_model_ids = {
        model.model_id for model in enabled_models
    }

    if default_model_id not in enabled_model_ids:
        raise ValueError(
            "Model registry defaultModelId must reference "
            "an enabled model."
        )

    for model in enabled_models:
        if not model.artifact_path.is_dir():
            raise NotADirectoryError(
                "Enabled model artifact directory does not exist: "
                f"{model.artifact_path}"
            )

    return ModelRegistryConfiguration(
        default_model_id=default_model_id,
        models=models,
    )


def _parse_model_entry(
    value: Any,
    index: int,
    registry_root: Path,
) -> ModelRegistryEntry:
    if not isinstance(value, dict):
        raise ValueError(
            f"Model registry entry {index} must be a JSON object."
        )

    _require_exact_keys(
        value,
        {
            "id",
            "displayName",
            "artifactDirectory",
            "enabled",
        },
        f"Model registry entry {index}",
    )

    model_id = _validate_model_id(
        value["id"],
        f"models[{index}].id",
    )
    display_name = _require_non_empty_string(
        value["displayName"],
        f"models[{index}].displayName",
    )
    artifact_directory = _require_non_empty_string(
        value["artifactDirectory"],
        f"models[{index}].artifactDirectory",
    )

    enabled = value["enabled"]

    if type(enabled) is not bool:
        raise ValueError(
            f"models[{index}].enabled must be a Boolean."
        )

    relative_artifact_path = Path(artifact_directory)

    if (
        relative_artifact_path.is_absolute()
        or relative_artifact_path.drive
    ):
        raise ValueError(
            f"models[{index}].artifactDirectory "
            "must be a relative path."
        )

    artifact_path = (
        registry_root / relative_artifact_path
    ).resolve()

    if not artifact_path.is_relative_to(registry_root):
        raise ValueError(
            f"models[{index}].artifactDirectory "
            "must stay inside the registry directory."
        )

    return ModelRegistryEntry(
        model_id=model_id,
        display_name=display_name,
        artifact_path=artifact_path,
        enabled=enabled,
    )


def _validate_model_id(
    value: Any,
    field_name: str,
) -> str:
    model_id = _require_non_empty_string(
        value,
        field_name,
    )

    if _MODEL_ID_PATTERN.fullmatch(model_id) is None:
        raise ValueError(
            f"{field_name} must contain only lowercase letters, "
            "digits, periods, underscores, and hyphens."
        )

    return model_id


def _require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def _require_exact_keys(
    value: dict[str, Any],
    expected_keys: set[str],
    context: str,
) -> None:
    actual_keys = set(value)

    if actual_keys != expected_keys:
        missing_keys = sorted(expected_keys - actual_keys)
        unexpected_keys = sorted(actual_keys - expected_keys)

        raise ValueError(
            f"{context} has invalid fields. "
            f"Missing: {missing_keys}. "
            f"Unexpected: {unexpected_keys}."
        )