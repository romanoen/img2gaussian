"""Small orchestration helpers for the pipeline stages."""

from __future__ import annotations

from pathlib import Path

from .colmap import run_colmap
from .config import AppConfig
from .preprocess import extract_frames, select_frames
from .render import run_rendering
from .train import run_training


def run_extract_stage(config: AppConfig) -> list[Path]:
    """Run only the frame extraction step."""

    return extract_frames(config)


def run_select_stage(config: AppConfig) -> list[Path]:
    """Run only the frame selection step."""

    return select_frames(config)


def run_colmap_stage(config: AppConfig) -> None:
    """Build the COLMAP dataset from the selected frames."""

    run_colmap(config)


def run_train_and_render_stage(config: AppConfig) -> dict[str, Path]:
    """Train the model and immediately export its renders."""

    run_training(config)
    return run_rendering(config)


def run_full_pipeline(config: AppConfig) -> dict[str, Path]:
    """Execute the end-to-end workflow from video frames to demo assets."""

    extract_frames(config)
    select_frames(config)
    run_colmap(config)
    run_training(config)
    return run_rendering(config)
