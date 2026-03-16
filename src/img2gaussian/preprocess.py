"""Video frame extraction and curation helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig, build_workspace_paths
from .utils import clear_matching_files, ensure_binary, list_image_files, run_command


def extract_frames(config: AppConfig) -> list[Path]:
    """Extract evenly timed frames from the input video with ffmpeg."""

    config.ensure_input_video_exists()
    ensure_binary("ffmpeg")

    paths = build_workspace_paths(config)
    # Start fresh so reruns do not mix frames from different FPS settings.
    clear_matching_files(paths.raw_frames_dir, "frame_*")

    output_pattern = paths.raw_frames_dir / "frame_%05d.png"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(config.input_video),
        "-vf",
        f"fps={config.fps}",
        str(output_pattern),
    ]
    run_command(command)

    frames = list_image_files(paths.raw_frames_dir)
    if not frames:
        raise RuntimeError("Frame extraction produced no images.")

    print(f"Extracted {len(frames)} frames into {paths.raw_frames_dir}")
    return frames


def select_frames(config: AppConfig) -> list[Path]:
    """Keep a sharp, evenly distributed subset of frames for reconstruction."""

    paths = build_workspace_paths(config)
    raw_frames = list_image_files(paths.raw_frames_dir)
    if not raw_frames:
        raise FileNotFoundError(
            f"No raw frames found in {paths.raw_frames_dir}. Run extract_frames first."
        )

    clear_matching_files(paths.selected_frames_dir, "frame_*")

    if len(raw_frames) <= config.max_frames:
        chosen_frames = raw_frames
    else:
        # Splitting first keeps coverage across the whole clip instead of clustering
        # around whichever span happens to have the sharpest images.
        frame_groups = np.array_split(np.array(raw_frames, dtype=object), config.max_frames)
        chosen_frames = [_pick_sharpest_frame(list(group)) for group in frame_groups if len(group)]

    selected_paths: list[Path] = []
    for index, frame_path in enumerate(chosen_frames, start=1):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        resized = _resize_to_long_side(image, config.max_long_side)
        # JPG is plenty here and keeps the downstream COLMAP set lighter on disk.
        output_path = paths.selected_frames_dir / f"frame_{index:05d}.jpg"
        ok = cv2.imwrite(str(output_path), resized)
        if not ok:
            raise RuntimeError(f"Could not write selected frame: {output_path}")
        selected_paths.append(output_path)

    print(
        f"Selected {len(selected_paths)} sharp frames into {paths.selected_frames_dir} "
        f"(max long side: {config.max_long_side}px)"
    )
    return selected_paths


def _pick_sharpest_frame(frame_paths: list[Path]) -> Path:
    """Choose the frame with the strongest Laplacian variance."""

    scored_frames = [(_blur_score(frame_path), frame_path) for frame_path in frame_paths]
    scored_frames.sort(key=lambda item: item[0], reverse=True)
    return scored_frames[0][1]


def _blur_score(frame_path: Path) -> float:
    """Use a simple focus metric to score how sharp a frame looks."""

    image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not score frame: {frame_path}")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _resize_to_long_side(image: np.ndarray, max_long_side: int) -> np.ndarray:
    """Downscale an image so its longest edge stays under the requested limit."""

    height, width = image.shape[:2]
    current_long_side = max(height, width)
    if current_long_side <= max_long_side:
        return image

    scale = max_long_side / float(current_long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
