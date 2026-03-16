"""CLI entrypoint for the COLMAP stage."""

from __future__ import annotations

from _shared import build_parser, ensure_src_on_path


def main() -> None:
    """Run the reconstruction step against the selected frames."""

    parser = build_parser("Run the COLMAP reconstruction stage.")
    args = parser.parse_args()

    ensure_src_on_path()

    from img2gaussian.config import load_config
    from img2gaussian.pipeline import run_colmap_stage

    config = load_config(args.config)
    run_colmap_stage(config)


if __name__ == "__main__":
    main()
