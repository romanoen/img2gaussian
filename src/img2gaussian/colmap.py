"""COLMAP reconstruction stage for turning selected frames into a dataset."""

from __future__ import annotations

from .config import AppConfig, WorkspacePaths, build_workspace_paths
from .utils import ensure_binary, list_image_files, run_command, safe_reset_dir


def run_colmap(config: AppConfig) -> None:
    """Run the COLMAP steps needed by the Gaussian Splatting training code."""

    ensure_binary("colmap")
    paths = build_workspace_paths(config)

    selected_frames = list_image_files(paths.selected_frames_dir)
    if not selected_frames:
        raise FileNotFoundError(
            f"No selected frames found in {paths.selected_frames_dir}. Run select_frames first."
        )

    safe_reset_dir(paths.colmap_dir)
    safe_reset_dir(paths.dataset_dir)
    paths.distorted_sparse_dir.mkdir(parents=True, exist_ok=True)

    # The baseline pipeline expects a single-camera sequence and a COLMAP-style
    # dataset folder, so we keep the commands fairly conservative here.
    feature_command = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(paths.colmap_database_path),
        "--image_path",
        str(paths.selected_frames_dir),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu",
        "0",
    ]
    matcher_command = [
        "colmap",
        "sequential_matcher",
        "--database_path",
        str(paths.colmap_database_path),
        "--SiftMatching.use_gpu",
        "0",
    ]
    mapper_command = [
        "colmap",
        "mapper",
        "--database_path",
        str(paths.colmap_database_path),
        "--image_path",
        str(paths.selected_frames_dir),
        "--output_path",
        str(paths.distorted_sparse_dir),
    ]
    undistort_command = [
        "colmap",
        "image_undistorter",
        "--image_path",
        str(paths.selected_frames_dir),
        "--input_path",
        str(paths.distorted_model_dir),
        "--output_path",
        str(paths.dataset_dir),
        "--output_type",
        "COLMAP",
    ]

    run_command(feature_command)
    run_command(matcher_command)
    run_command(mapper_command)

    if not paths.distorted_model_dir.exists():
        raise RuntimeError(
            "COLMAP did not produce a sparse model. "
            "Try a steadier video or keep more background texture visible."
        )

    run_command(undistort_command)

    _normalize_sparse_directory(paths)

    if not paths.dataset_sparse_dir.exists():
        raise RuntimeError("COLMAP undistortion did not create dataset/sparse/0.")
    if not list_image_files(paths.dataset_images_dir):
        raise RuntimeError("COLMAP undistortion did not create any dataset images.")

    print(f"COLMAP dataset ready at {paths.dataset_dir}")


def _normalize_sparse_directory(paths: WorkspacePaths) -> None:
    """Handle both nested and flat sparse-directory layouts from COLMAP."""

    sparse_root = paths.dataset_dir / "sparse"
    if paths.dataset_sparse_dir.exists():
        return

    flat_sparse_files = [
        *sorted(sparse_root.glob("*.bin")),
        *sorted(sparse_root.glob("*.txt")),
    ]
    if not flat_sparse_files:
        return

    paths.dataset_sparse_dir.mkdir(parents=True, exist_ok=True)
    for source_path in flat_sparse_files:
        source_path.rename(paths.dataset_sparse_dir / source_path.name)
