from __future__ import annotations

from pathlib import Path

from .colmap import run_colmap
from .config import AppConfig
from .preprocess import extract_frames, select_frames
from .render import run_rendering
from .train import run_training


def run_extract_stage(config: AppConfig) -> list[Path]:
    return extract_frames(config)


def run_select_stage(config: AppConfig) -> list[Path]:
    return select_frames(config)


def run_colmap_stage(config: AppConfig) -> None:
    run_colmap(config)


def run_train_and_render_stage(config: AppConfig) -> dict[str, Path]:
    run_training(config)
    return run_rendering(config)


def run_full_pipeline(config: AppConfig) -> dict[str, Path]:
    extract_frames(config)
    select_frames(config)
    run_colmap(config)
    run_training(config)
    return run_rendering(config)
