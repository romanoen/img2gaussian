# img2gaussian

`img2gaussian` is a small, readable pipeline for turning a single handheld video into a static 3D Gaussian scene.

The main idea is simple:

```text
video
  -> extract frames
  -> keep a sharp, evenly spaced subset
  -> estimate camera poses with COLMAP
  -> train a Gaussian Splatting model
  -> export stills, a demo clip, and a browser viewer
```

This repository is meant to be understandable first. It uses strong external tools where that makes sense, and keeps the project-specific logic in a thin orchestration layer that is easy to read.

## What this project does

Given one input video, the pipeline:

- extracts frames with `ffmpeg`
- filters them down to a cleaner training subset
- reconstructs camera poses with `COLMAP`
- trains a 3D Gaussian scene using the GraphDeco Gaussian Splatting baseline
- exports renders and a browser-based viewer

The result is a static scene representation that can be viewed from new camera angles.

## Design goals

The project is built around a few practical goals:

- keep the code small and readable
- make the full pipeline reproducible
- use one config file per run
- preserve useful intermediate outputs
- make the final result easy to inspect locally or over SSH

Instead of reimplementing SfM or Gaussian Splatting, this repo focuses on the glue around them:

- input preparation
- configuration
- staging the pipeline
- organizing outputs
- viewer export

## Repository layout

- `configs/`
  YAML presets for different run qualities.
- `scripts/`
  Small CLI entrypoints for setup and pipeline stages.
- `src/img2gaussian/`
  The current Python package for the pipeline logic.
- `third_party/gaussian-splatting/`
  Pinned upstream Gaussian Splatting baseline.
- `data/input/`
  Input videos.
- `outputs/`
  Generated frames, reconstructions, models, renders, and viewer exports.

## Environment

This project uses `micromamba`.

Create the environment:

```bash
micromamba env create -f environment.yml -n face-gaussian
micromamba activate face-gaussian
```

The environment includes the stable command-line dependencies:

- Python 3.10
- `ffmpeg`
- `colmap`
- CUDA 11.8 toolchain packages
- `cmake`
- `ninja`
- `git`

Install PyTorch separately so the wheel matches the local CUDA setup:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Setup

After the environment is active, run these two scripts.

Bootstrap the upstream Gaussian Splatting repo:

```bash
python scripts/bootstrap.py
```

Install the Python-side Gaussian dependencies:

```bash
python scripts/install_gaussian_deps.py
```

What those scripts do:

- verify required binaries like `git`, `ffmpeg`, and `colmap`
- clone and pin the upstream Gaussian Splatting repository
- install `plyfile` and `joblib`
- build the required CUDA extensions
- verify that CUDA is available to PyTorch

## Configuration

The pipeline is controlled through YAML config files.

Included presets:

- [`configs/default.yaml`](configs/default.yaml)
- [`configs/high_quality.yaml`](configs/high_quality.yaml)

Important config fields:

- `input_video`
- `workspace_dir`
- `fps`
- `max_frames`
- `max_long_side`
- `train_iterations`
- `render_mode`
- `gaussian_repo_dir`

`default.yaml` is the safer preset for smaller GPUs.  
`high_quality.yaml` pushes quality a bit further at the cost of speed and memory.

## Input capture

This pipeline works best when the input video is recorded like a reconstruction capture instead of a casual clip.

Recommended:

- record for about `10-30` seconds
- move slowly
- keep lighting stable
- avoid heavy motion blur
- keep enough scene texture visible for COLMAP

In practice, this works best on scenes that are mostly static while the video is being recorded.

## Running the pipeline

Run the full pipeline:

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

Or run each stage separately:

```bash
python scripts/extract_frames.py --config configs/default.yaml
python scripts/select_frames.py --config configs/default.yaml
python scripts/run_colmap.py --config configs/default.yaml
python scripts/train_and_render.py --config configs/default.yaml
```

Running stage-by-stage is useful when checking a new video, because you can inspect the selected frames and the COLMAP reconstruction before spending time on training.

## Stage breakdown

### `extract_frames.py`

Uses `ffmpeg` to write raw frames into `frames_raw/`.

### `select_frames.py`

Scores sharpness, keeps a cleaner subset, resizes images, and writes them into `frames_selected/`.

### `run_colmap.py`

Runs the COLMAP steps used to estimate camera poses and prepare the dataset structure expected by the trainer.

### `train_and_render.py`

Launches Gaussian Splatting training and then exports still images plus a short demo video.

## Interactive viewer

The project includes a browser-based Gaussian viewer export.

Build the viewer:

```bash
python scripts/build_browser_viewer.py --config configs/high_quality.yaml
```

Build and serve it locally:

```bash
python scripts/run_browser_viewer.py --config configs/high_quality.yaml --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

This viewer path works well over SSH port forwarding or Tailscale because it only needs a local HTTP server.

## Outputs

Each run writes into the configured `workspace_dir`.

Typical outputs:

- `frames_raw/`
- `frames_selected/`
- `colmap/`
- `dataset/`
- `model/`
- `renders/stills/`
- `renders/demo.mp4`
- `browser_gaussian_viewer/index.html`

## Notes

- If `colmap` in the environment is unstable on your machine, you can use a system `colmap` binary and keep the same project code.
- If CUDA is not available, reinstall the PyTorch wheel inside the active environment.
- If training runs out of memory, reduce `max_long_side` and `train_iterations`.

## Why this repo exists

The point of this repository is to make the path from raw video to a usable 3D Gaussian scene easy to follow.

The value here is that the full workflow stays visible:

- where the data comes from
- how it is filtered
- how reconstruction is staged
- how training is launched
- where the final artifacts end up
