"""Install or verify the Python pieces required by Gaussian Splatting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from _shared import build_parser, ensure_src_on_path


def main() -> None:
    """Install the missing runtime and extension dependencies into the active env."""

    parser = build_parser(
        "Install or verify the Gaussian Splatting Python dependencies in the active environment."
    )
    args = parser.parse_args()

    ensure_src_on_path()

    from img2gaussian.config import load_config
    from img2gaussian.utils import python_executable, run_command

    config = load_config(args.config)
    repo_dir = config.gaussian_repo_dir
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Gaussian repo not found at {repo_dir}. Run python scripts/bootstrap.py first."
        )

    required_modules = (
        "torch",
        "plyfile",
        "joblib",
        "diff_gaussian_rasterization",
        "simple_knn",
        "fused_ssim",
    )
    missing_modules = [
        module_name
        for module_name in required_modules
        if importlib.util.find_spec(module_name) is None
    ]

    if importlib.util.find_spec("torch") is None:
        run_command(
            [
                python_executable(),
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cu118",
            ]
        )

    if missing_modules:
        # The custom CUDA extensions build more reliably when we point them at the
        # active conda-style toolchain and CUDA headers explicitly.
        env = _build_extension_env(Path(sys.prefix))
        run_command(
            [
                python_executable(),
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "plyfile",
                "joblib",
                str(repo_dir / "submodules" / "diff-gaussian-rasterization"),
                str(repo_dir / "submodules" / "simple-knn"),
                str(repo_dir / "submodules" / "fused-ssim"),
            ],
            env=env,
        )

    _verify_runtime()
    print("Gaussian Python dependencies are ready.")


def _build_extension_env(prefix: Path) -> dict[str, str]:
    """Construct the environment variables expected by the CUDA extension builds."""

    cuda_include = prefix / "targets" / "x86_64-linux" / "include"
    cuda_lib = prefix / "targets" / "x86_64-linux" / "lib"
    cc = prefix / "bin" / "x86_64-conda-linux-gnu-cc"
    cxx = prefix / "bin" / "x86_64-conda-linux-gnu-c++"

    return {
        "CUDA_HOME": str(prefix),
        "CC": str(cc),
        "CXX": str(cxx),
        "CUDAHOSTCXX": str(cxx),
        "CPATH": str(cuda_include),
        "CPLUS_INCLUDE_PATH": str(cuda_include),
        "LIBRARY_PATH": f"{prefix / 'lib'}:{cuda_lib}",
        "TORCH_CUDA_ARCH_LIST": "6.1",
    }


def _verify_runtime() -> None:
    """Confirm that the installed PyTorch build can access CUDA."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch was installed, but torch.cuda.is_available() is False. "
            "Check your driver and CUDA wheel selection."
        )


if __name__ == "__main__":
    main()
