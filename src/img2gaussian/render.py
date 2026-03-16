"""Rendering helpers plus small demo-asset generation utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig, WorkspacePaths, build_workspace_paths
from .utils import (
    CommandError,
    copy_file,
    ensure_binary,
    list_image_files,
    python_executable,
    run_command,
    safe_reset_dir,
)

VIDEO_FILTER = (
    "scale=1280:720:force_original_aspect_ratio=decrease,"
    "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,fps=24"
)


def run_rendering(config: AppConfig) -> dict[str, Path]:
    """Render the trained model and assemble the lightweight demo outputs."""

    config.ensure_gaussian_repo_exists()
    ensure_binary("ffmpeg")

    paths = build_workspace_paths(config)
    render_script = config.gaussian_repo_dir / "render.py"
    if not render_script.is_file():
        raise FileNotFoundError(f"Expected render.py at {render_script}")

    command = [
        python_executable(),
        str(render_script),
        "-m",
        str(paths.model_dir),
        "-s",
        str(paths.dataset_dir),
        "--iteration",
        str(config.train_iterations),
    ]
    if config.render_mode == "novel_views":
        command.append("--skip_train")
    if config.antialiasing:
        command.append("--antialiasing")

    try:
        run_command(command)
    except CommandError as exc:
        raise RuntimeError("Gaussian rendering failed. Inspect the terminal output above.") from exc

    render_frames = _collect_render_frames(paths, config.train_iterations, config.render_mode)
    if not render_frames:
        raise RuntimeError("Rendering produced no images to export.")

    safe_reset_dir(paths.stills_dir)
    still_paths = _export_stills(render_frames, paths.stills_dir, count=3)
    _build_demo_video(config, paths, render_frames, still_paths[0])

    print(f"Rendered {len(render_frames)} frames into {paths.renders_dir}")
    print(f"Exported stills into {paths.stills_dir}")
    print(f"Demo video written to {paths.demo_video_path}")
    return {"demo_video": paths.demo_video_path, "stills_dir": paths.stills_dir}


def _collect_render_frames(
    paths: WorkspacePaths,
    iteration: int,
    render_mode: str,
) -> list[Path]:
    """Find the render frames emitted by the upstream script."""

    test_dir = paths.model_dir / "test" / f"ours_{iteration}" / "renders"
    train_dir = paths.model_dir / "train" / f"ours_{iteration}" / "renders"

    if render_mode == "novel_views":
        frames = list_image_files(test_dir)
        if frames:
            return frames
        return list_image_files(train_dir)

    frames = list_image_files(test_dir)
    frames.extend(list_image_files(train_dir))
    return frames


def _export_stills(render_frames: list[Path], output_dir: Path, count: int) -> list[Path]:
    """Copy a few representative render frames into a stable output folder."""

    still_paths: list[Path] = []
    for index, frame_path in enumerate(render_frames[:count], start=1):
        target_path = output_dir / f"still_{index:02d}{frame_path.suffix.lower()}"
        copy_file(frame_path, target_path)
        still_paths.append(target_path)
    return still_paths


def _build_demo_video(
    config: AppConfig,
    paths: WorkspacePaths,
    render_frames: list[Path],
    hero_still: Path,
) -> None:
    """Assemble a short presentation video from the intermediate artifacts."""

    ensure_binary("ffmpeg")
    safe_reset_dir(paths.demo_assets_dir)

    frame_grid = paths.demo_assets_dir / "selected_grid.png"
    _write_contact_sheet(
        list_image_files(paths.selected_frames_dir)[:12],
        frame_grid,
        columns=4,
    )

    input_segment = paths.demo_assets_dir / "segment_01_input.mp4"
    frames_segment = paths.demo_assets_dir / "segment_02_frames.mp4"
    recon_segment = paths.demo_assets_dir / "segment_03_reconstruction.mp4"
    novel_segment = paths.demo_assets_dir / "segment_04_novel_views.mp4"

    # The sequence is intentionally simple: input clip, frame grid, a hero still,
    # then the rendered sweep.
    _build_input_segment(config.input_video, input_segment)
    _build_still_segment(frame_grid, frames_segment, duration_seconds=2)
    _build_still_segment(hero_still, recon_segment, duration_seconds=2)
    _build_novel_view_segment(render_frames, novel_segment)

    segment_manifest = paths.demo_assets_dir / "segments.txt"
    with segment_manifest.open("w", encoding="utf-8") as handle:
        for segment in (input_segment, frames_segment, recon_segment, novel_segment):
            handle.write(f"file '{segment.as_posix()}'\n")

    concat_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(segment_manifest),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(paths.demo_video_path),
    ]
    run_command(concat_command)


def _build_input_segment(input_video: Path, output_video: Path) -> None:
    """Trim and normalize the source video into the first demo segment."""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(input_video),
        "-t",
        "4",
        "-vf",
        VIDEO_FILTER,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    run_command(command)


def _build_still_segment(image_path: Path, output_video: Path, duration_seconds: int) -> None:
    """Turn a single image into a fixed-length video segment."""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        str(duration_seconds),
        "-vf",
        VIDEO_FILTER,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    run_command(command)


def _build_novel_view_segment(render_frames: list[Path], output_video: Path) -> None:
    """Encode the rendered frames as a short fly-around segment."""

    manifest_path = output_video.parent / "novel_views_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for frame_path in render_frames:
            handle.write(f"file '{frame_path.as_posix()}'\n")
            handle.write("duration 0.12\n")
        # Repeating the last frame keeps ffmpeg's concat demuxer from dropping it.
        handle.write(f"file '{render_frames[-1].as_posix()}'\n")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest_path),
        "-vf",
        VIDEO_FILTER,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    run_command(command)


def _write_contact_sheet(image_paths: list[Path], output_path: Path, columns: int) -> None:
    """Lay out a small grid of selected frames for the demo video."""

    if not image_paths:
        raise RuntimeError("Cannot build a frame grid without selected frames.")

    cell_width = 320
    cell_height = 180
    rows = int(np.ceil(len(image_paths) / columns))
    canvas = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)

    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read selected frame: {image_path}")

        fitted = _fit_image_to_cell(image, cell_width, cell_height)
        row = index // columns
        col = index % columns
        # Center each fitted image so mismatched aspect ratios still look tidy.
        y0 = row * cell_height + (cell_height - fitted.shape[0]) // 2
        x0 = col * cell_width + (cell_width - fitted.shape[1]) // 2
        canvas[y0 : y0 + fitted.shape[0], x0 : x0 + fitted.shape[1]] = fitted

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def _fit_image_to_cell(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to fit inside a contact-sheet cell."""

    scale = min(width / image.shape[1], height / image.shape[0])
    new_width = max(1, int(round(image.shape[1] * scale)))
    new_height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
