from __future__ import annotations

from _shared import build_parser, ensure_src_on_path


def main() -> None:
    parser = build_parser("Run the full face Gaussian pipeline from video to demo render.")
    args = parser.parse_args()

    ensure_src_on_path()

    from face_gaussian.config import load_config
    from face_gaussian.pipeline import run_full_pipeline

    config = load_config(args.config)
    run_full_pipeline(config)


if __name__ == "__main__":
    main()
