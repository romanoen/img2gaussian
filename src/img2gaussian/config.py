"""Configuration loading and workspace path conventions for the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_RENDER_MODES = {"novel_views", "all_views"}
ALLOWED_DATA_DEVICES = {"cuda", "cpu"}


@dataclass(frozen=True)
class AppConfig:
    """Normalized application settings loaded from a YAML file."""

    input_video: Path
    workspace_dir: Path
    fps: float
    max_frames: int
    max_long_side: int
    train_iterations: int
    render_mode: str
    gaussian_repo_dir: Path
    antialiasing: bool
    data_device: str

    def ensure_input_video_exists(self) -> None:
        """Fail early when the configured input video is missing."""

        if not self.input_video.is_file():
            raise FileNotFoundError(
                f"Input video not found: {self.input_video}. "
                "Update input_video in your config."
            )

    def ensure_gaussian_repo_exists(self) -> None:
        """Make sure the baseline Gaussian Splatting checkout is available."""

        train_script = self.gaussian_repo_dir / "train.py"
        if not train_script.is_file():
            raise FileNotFoundError(
                "Gaussian Splatting baseline not found. "
                f"Expected train.py at {train_script}. "
                "Run python scripts/bootstrap.py first."
            )


@dataclass(frozen=True)
class WorkspacePaths:
    """Canonical filesystem layout derived from a single workspace directory."""

    workspace_dir: Path
    raw_frames_dir: Path
    selected_frames_dir: Path
    colmap_dir: Path
    colmap_database_path: Path
    distorted_sparse_dir: Path
    distorted_model_dir: Path
    dataset_dir: Path
    dataset_images_dir: Path
    dataset_sparse_dir: Path
    model_dir: Path
    renders_dir: Path
    stills_dir: Path
    demo_assets_dir: Path
    demo_video_path: Path


def load_config(config_path: str | Path) -> AppConfig:
    """Read, validate, and normalize a project configuration file."""

    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    required_keys = {
        "input_video",
        "workspace_dir",
        "fps",
        "max_frames",
        "max_long_side",
        "train_iterations",
        "render_mode",
        "gaussian_repo_dir",
    }
    missing = sorted(required_keys - raw.keys())
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Config is missing required keys: {missing_keys}")

    render_mode = str(raw["render_mode"]).strip()
    if render_mode not in ALLOWED_RENDER_MODES:
        allowed = ", ".join(sorted(ALLOWED_RENDER_MODES))
        raise ValueError(f"render_mode must be one of: {allowed}")

    data_device = str(raw.get("data_device", "cuda")).strip()
    if data_device not in ALLOWED_DATA_DEVICES:
        allowed = ", ".join(sorted(ALLOWED_DATA_DEVICES))
        raise ValueError(f"data_device must be one of: {allowed}")

    fps = float(raw["fps"])
    max_frames = int(raw["max_frames"])
    max_long_side = int(raw["max_long_side"])
    train_iterations = int(raw["train_iterations"])
    antialiasing = bool(raw.get("antialiasing", False))

    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than 0")
    if max_long_side < 256:
        raise ValueError("max_long_side must be at least 256")
    if train_iterations <= 0:
        raise ValueError("train_iterations must be greater than 0")

    return AppConfig(
        input_video=_resolve_project_path(raw["input_video"]),
        workspace_dir=_resolve_project_path(raw["workspace_dir"]),
        fps=fps,
        max_frames=max_frames,
        max_long_side=max_long_side,
        train_iterations=train_iterations,
        render_mode=render_mode,
        gaussian_repo_dir=_resolve_project_path(raw["gaussian_repo_dir"]),
        antialiasing=antialiasing,
        data_device=data_device,
    )


def build_workspace_paths(config: AppConfig) -> WorkspacePaths:
    """Expand the configured workspace root into the directories each stage uses."""

    workspace_dir = config.workspace_dir
    return WorkspacePaths(
        workspace_dir=workspace_dir,
        raw_frames_dir=workspace_dir / "frames_raw",
        selected_frames_dir=workspace_dir / "frames_selected",
        colmap_dir=workspace_dir / "colmap",
        colmap_database_path=workspace_dir / "colmap" / "database.db",
        distorted_sparse_dir=workspace_dir / "colmap" / "distorted" / "sparse",
        distorted_model_dir=workspace_dir / "colmap" / "distorted" / "sparse" / "0",
        dataset_dir=workspace_dir / "dataset",
        dataset_images_dir=workspace_dir / "dataset" / "images",
        dataset_sparse_dir=workspace_dir / "dataset" / "sparse" / "0",
        model_dir=workspace_dir / "model",
        renders_dir=workspace_dir / "renders",
        stills_dir=workspace_dir / "renders" / "stills",
        demo_assets_dir=workspace_dir / "renders" / "demo_assets",
        demo_video_path=workspace_dir / "renders" / "demo.mp4",
    )


def _resolve_project_path(value: str) -> Path:
    """Resolve relative config paths against the repository root."""

    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()
