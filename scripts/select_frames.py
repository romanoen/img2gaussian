from __future__ import annotations

from _shared import build_parser, ensure_src_on_path


def main() -> None:
    parser = build_parser("Select sharp, evenly spaced frames for COLMAP and training.")
    args = parser.parse_args()

    ensure_src_on_path()

    from face_gaussian.config import load_config
    from face_gaussian.pipeline import run_select_stage

    config = load_config(args.config)
    run_select_stage(config)


if __name__ == "__main__":
    main()
