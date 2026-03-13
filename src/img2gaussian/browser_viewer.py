from __future__ import annotations

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import AppConfig, build_workspace_paths
from .utils import ensure_binary, find_latest_point_cloud, run_command

DEFAULT_PORT = 8765
DEFAULT_COMPRESSION_ITERATIONS = 1
DEFAULT_HARMONIC_BANDS = 1


def ensure_browser_viewer_dependencies(project_root: Path) -> Path:
    ensure_binary("npm")
    viewer_tool_dir = project_root / "browser_viewer"
    package_json = viewer_tool_dir / "package.json"
    if not package_json.is_file():
        raise FileNotFoundError(f"Browser viewer package.json not found: {package_json}")

    binary = viewer_tool_dir / "node_modules" / ".bin" / "splat-transform"
    if binary.is_file():
        return binary

    run_command(["npm", "install", "@playcanvas/splat-transform"], cwd=viewer_tool_dir)
    if not binary.is_file():
        raise FileNotFoundError(
            "splat-transform was not installed successfully. "
            f"Expected binary at {binary}."
        )
    return binary


def build_browser_viewer(config: AppConfig, project_root: Path) -> Path:
    binary = ensure_browser_viewer_dependencies(project_root)
    paths = build_workspace_paths(config)
    point_cloud_path = find_latest_point_cloud(paths.model_dir)

    viewer_dir = paths.workspace_dir / "browser_gaussian_viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    output_html = viewer_dir / "index.html"

    command = [
        str(binary),
        "-w",
        "-g",
        "cpu",
        "-i",
        str(DEFAULT_COMPRESSION_ITERATIONS),
        str(point_cloud_path),
        "-H",
        str(DEFAULT_HARMONIC_BANDS),
        str(output_html),
    ]
    run_command(command, cwd=project_root / "browser_viewer")
    return output_html


def serve_browser_viewer(viewer_html_path: Path, port: int = DEFAULT_PORT) -> None:
    if not viewer_html_path.is_file():
        raise FileNotFoundError(f"Browser viewer HTML not found: {viewer_html_path}")
    directory = viewer_html_path.parent
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Viewer assets: {directory}")
    print(f"Open {url} in your browser.")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
