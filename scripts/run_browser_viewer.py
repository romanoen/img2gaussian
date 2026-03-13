from __future__ import annotations

from _shared import PROJECT_ROOT, build_parser, ensure_src_on_path


def main() -> None:
    parser = build_parser("Build and serve the browser-based Gaussian viewer.")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port used to serve the browser viewer.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Serve an existing browser viewer build without rebuilding it first.",
    )
    args = parser.parse_args()

    ensure_src_on_path()

    from img2gaussian.browser_viewer import build_browser_viewer, serve_browser_viewer
    from img2gaussian.config import build_workspace_paths, load_config

    config = load_config(args.config)
    if args.skip_build:
        viewer_html = build_workspace_paths(config).workspace_dir / "browser_gaussian_viewer" / "index.html"
    else:
        viewer_html = build_browser_viewer(config, project_root=PROJECT_ROOT)
    serve_browser_viewer(viewer_html, port=args.port)


if __name__ == "__main__":
    main()
