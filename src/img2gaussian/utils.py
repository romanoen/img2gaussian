"""Shared subprocess, filesystem, and discovery helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
import os
from pathlib import Path


class CommandError(RuntimeError):
    """Raised when an external command exits with a non-zero status."""


def ensure_binary(name: str) -> str:
    """Return the absolute path to a required CLI tool."""

    binary = shutil.which(name)
    if not binary:
        raise FileNotFoundError(
            f"Required binary '{name}' was not found in PATH. "
            "Activate the micromamba environment or install the tool first."
        )
    return binary


def run_command(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run an external command while echoing it for easier debugging."""

    printable = " ".join(command)
    print(f"[cmd] {printable}")
    merged_env = os.environ.copy()
    # Let callers patch in build-specific variables without losing the parent env.
    if env:
        merged_env.update(env)
    completed = subprocess.run(command, cwd=cwd, env=merged_env, check=False)
    if completed.returncode != 0:
        raise CommandError(
            f"Command failed with exit code {completed.returncode}: {printable}"
        )


def safe_reset_dir(path: Path) -> None:
    """Recreate a directory from scratch."""

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clear_matching_files(directory: Path, pattern: str) -> None:
    """Delete files matching a glob without removing the directory itself."""

    directory.mkdir(parents=True, exist_ok=True)
    for item in directory.glob(pattern):
        if item.is_file():
            item.unlink()


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file while creating the destination parent directory if needed."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def find_latest_point_cloud(model_dir: Path) -> Path:
    """Pick the point cloud from the highest available training iteration."""

    point_cloud_root = model_dir / "point_cloud"
    candidates = sorted(point_cloud_root.glob("iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(
            f"No trained point cloud found under {point_cloud_root}. "
            "Run train_and_render first."
        )
    return max(candidates, key=_iteration_number)


def list_image_files(directory: Path) -> list[Path]:
    """Collect supported image files in a stable order."""

    patterns = ("*.png", "*.jpg", "*.jpeg")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(directory.glob(pattern)))
    return sorted(files)


def python_executable() -> str:
    """Return the Python interpreter that launched the current process."""

    return sys.executable


def _iteration_number(path: Path) -> int:
    """Extract the numeric iteration marker from a checkpoint path."""

    for part in path.parts:
        if part.startswith("iteration_"):
            return int(part.split("_", maxsplit=1)[1])
    return -1
