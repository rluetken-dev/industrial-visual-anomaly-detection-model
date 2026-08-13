from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabeledImage:
    """Describe one image and its expected anomaly label."""

    path: Path
    group: str
    is_anomalous: bool