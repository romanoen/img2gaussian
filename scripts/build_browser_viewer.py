from __future__ import annotations

from _shared import PROJECT_ROOT, build_parser, ensure_src_on_path


def main() -> None:
    parser = build_parser("Build a browser-based Gaussian viewer for the latest trained model.")
    args = parser.parse_args()

    ensure_src_on_path()

    from face_gaussian.browser_viewer import build_browser_viewer
    from face_gaussian.config import load_config

    config = load_config(args.config)
    output_html = build_browser_viewer(config, project_root=PROJECT_ROOT)
    print(f"Browser viewer written to {output_html}")


if __name__ == "__main__":
    main()
