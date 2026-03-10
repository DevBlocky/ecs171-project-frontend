from __future__ import annotations

from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def artifact_path(name: str) -> Path:
    return ARTIFACTS_DIR / name
