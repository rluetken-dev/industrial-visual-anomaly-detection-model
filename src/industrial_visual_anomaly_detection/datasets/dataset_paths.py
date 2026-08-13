from pathlib import Path


def resolve_dataset_image_paths(
    dataset_root: Path,
    relative_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Resolve and validate image paths below a local dataset root."""

    resolved_dataset_root = dataset_root.resolve()

    if not resolved_dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist: {resolved_dataset_root}"
        )

    resolved_paths: list[Path] = []
    errors: list[str] = []

    for relative_path in relative_paths:
        resolved_path = (resolved_dataset_root / relative_path).resolve()

        if not resolved_path.is_relative_to(resolved_dataset_root):
            errors.append(
                f"Image path escapes the dataset root: {relative_path}"
            )
            continue

        if not resolved_path.is_file():
            errors.append(
                f"Dataset image does not exist: {resolved_path}"
            )
            continue

        resolved_paths.append(resolved_path)

    if errors:
        raise ValueError("\n".join(errors))

    return tuple(resolved_paths)