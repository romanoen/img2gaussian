"""CLI entrypoint for the frame extraction stage."""

from __future__ import annotations

from _shared import build_parser, ensure_src_on_path


def main() -> None:
    """Extract frames from the configured input video."""

    parser = build_parser("Extract frames from the configured input video.")
    args = parser.parse_args()

    ensure_src_on_path()

    from img2gaussian.config import load_config
    from img2gaussian.pipeline import run_extract_stage

    config = load_config(args.config)
    run_extract_stage(config)


if __name__ == "__main__":
    main()
