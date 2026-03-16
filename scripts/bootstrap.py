"""Bootstrap the upstream Gaussian Splatting checkout and basic prerequisites."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from _shared import build_parser, ensure_src_on_path

GAUSSIAN_REMOTE = "https://github.com/graphdeco-inria/gaussian-splatting.git"
GAUSSIAN_REF = "54c035f7834b564019656c3e3fcc3646292f727d"


def main() -> None:
    """Validate local tools and prepare the pinned Gaussian Splatting repo."""

    parser = build_parser("Check local tools and bootstrap the Gaussian Splatting baseline.")
    args = parser.parse_args()

    ensure_src_on_path()

    from img2gaussian.config import load_config
    from img2gaussian.utils import ensure_binary, run_command

    config = load_config(args.config)

    for binary in ("git", "ffmpeg", "colmap"):
        resolved = ensure_binary(binary)
        print(f"Found {binary}: {resolved}")

    repo_dir = config.gaussian_repo_dir
    # Keeping the upstream repo pinned makes the rest of this project much less
    # surprising to rerun a few weeks later.
    _clone_repo_if_needed(repo_dir, run_command)
    _checkout_repo_ref(repo_dir, run_command)
    _print_repo_status(repo_dir)
    _print_python_dependency_hints(repo_dir)


def _clone_repo_if_needed(repo_dir: Path, run_command) -> None:
    """Clone the upstream repository only when it is not already present."""

    if repo_dir.exists():
        git_dir = repo_dir / ".git"
        if not git_dir.exists():
            raise RuntimeError(
                f"{repo_dir} already exists but is not a git checkout. "
                "Remove it manually or point gaussian_repo_dir elsewhere."
            )
        print(f"Gaussian Splatting repo already exists at {repo_dir}")
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", "--recursive", GAUSSIAN_REMOTE, str(repo_dir)])


def _checkout_repo_ref(repo_dir: Path, run_command) -> None:
    """Move the repo to the exact commit this project was tested against."""

    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"Could not inspect git status for {repo_dir}")
    if status.stdout.strip():
        raise RuntimeError(
            f"{repo_dir} has local changes. Refusing to change the checkout automatically."
        )

    run_command(["git", "-C", str(repo_dir), "checkout", GAUSSIAN_REF])
    run_command(["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"])


def _print_repo_status(repo_dir: Path) -> None:
    """Echo the checked-out commit for a quick sanity check."""

    head = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode == 0:
        print(f"Gaussian Splatting checkout ready at commit {head.stdout.strip()}")


def _print_python_dependency_hints(repo_dir: Path) -> None:
    """Report missing Python modules without installing anything automatically."""

    missing_modules = []
    for module_name in (
        "torch",
        "plyfile",
        "joblib",
        "diff_gaussian_rasterization",
        "simple_knn",
        "fused_ssim",
    ):
        if importlib.util.find_spec(module_name) is None:
            missing_modules.append(module_name)

    if not missing_modules:
        print("Python-side Gaussian dependencies look available.")
        return

    print("Some Python dependencies are still missing:")
    for module_name in missing_modules:
        print(f"  - {module_name}")

    print("Recommended next steps:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("  python scripts/install_gaussian_deps.py")


if __name__ == "__main__":
    main()
