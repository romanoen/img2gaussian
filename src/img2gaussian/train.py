from __future__ import annotations

from .config import AppConfig, build_workspace_paths
from .utils import CommandError, python_executable, run_command


def run_training(config: AppConfig) -> None:
    config.ensure_gaussian_repo_exists()
    _ensure_torch_cuda()

    paths = build_workspace_paths(config)
    if not paths.dataset_sparse_dir.exists():
        raise FileNotFoundError(
            f"COLMAP dataset not found at {paths.dataset_sparse_dir}. Run run_colmap first."
        )

    train_script = config.gaussian_repo_dir / "train.py"
    command = [
        python_executable(),
        str(train_script),
        "-s",
        str(paths.dataset_dir),
        "-m",
        str(paths.model_dir),
        "--iterations",
        str(config.train_iterations),
        "--save_iterations",
        str(config.train_iterations),
        "--eval",
        "--data_device",
        config.data_device,
    ]
    if config.antialiasing:
        command.append("--antialiasing")

    try:
        run_command(command)
    except CommandError as exc:
        raise RuntimeError(
            "Gaussian training failed. "
            "If this was a CUDA OOM, lower max_long_side to 640 and train_iterations to 3500."
        ) from exc

    point_cloud_dir = paths.model_dir / "point_cloud" / f"iteration_{config.train_iterations}"
    if not point_cloud_dir.exists():
        raise RuntimeError(
            "Training finished without writing the expected point cloud directory. "
            "Check the Gaussian Splatting logs in the terminal output."
        )

    print(f"Training finished at {point_cloud_dir}")


def _ensure_torch_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install a CUDA build, for example:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. "
            "Install a CUDA-enabled PyTorch build in the active micromamba environment."
        )
