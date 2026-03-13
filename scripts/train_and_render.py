from __future__ import annotations

from _shared import build_parser, ensure_src_on_path


def main() -> None:
    parser = build_parser("Train the Gaussian model and export renders.")
    args = parser.parse_args()

    ensure_src_on_path()

    from img2gaussian.config import load_config
    from img2gaussian.pipeline import run_train_and_render_stage

    config = load_config(args.config)
    run_train_and_render_stage(config)


if __name__ == "__main__":
    main()
