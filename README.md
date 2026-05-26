# PKOD

PKOD is a parking and license plate recognition project built around:

- vehicle detection and tracking
- parking entry/exit counting
- ROI-based plate capture
- asynchronous OCR processing
- optional PostgreSQL logging
- research extensions for temporal OCR fusion on video

The repo is not a generic ALPR framework. It is an existing parking-gate pipeline where the main detector/tracker loop runs live, and OCR runs as a separate worker over saved OCR jobs.

It supports:

- recorded video files
- direct camera input by device index
- CCTV and network streams over RTSP/HTTP

## What The Repo Does

At a high level, the system works like this:

1. `PKOD1/main.py` opens a video file, camera feed, or CCTV/network stream, detects vehicles, tracks them, and counts entries/exits.
2. When a vehicle is stable inside the configured plate ROI, the system saves cropped frames into `ocr_jobs/`.
3. `PKOD1/ocr_processor.py` watches `ocr_jobs/`, detects the plate inside those saved frames, runs OCR, and stores the result.
4. Results are saved locally and can also be logged to PostgreSQL if `DATABASE_URL` is configured.
5. The research layer can fuse OCR across multiple frames instead of relying on one frame only.

## Main Repo Structure

```text
PKOD/
├── PKOD1/
│   ├── main.py                  # live vehicle tracking + counting loop
│   ├── ocr_processor.py         # async OCR worker
│   ├── admin_dashboard.py       # manual control GUI
│   ├── config.py                # runtime configuration
│   ├── capture/                 # video source handling
│   ├── tracking/                # detection + tracker config
│   ├── identity/                # track state / re-identification helpers
│   ├── roi_ocr/                 # ROI observer + OCR job creation
│   ├── ui/                      # overlays
│   ├── db/                      # PostgreSQL connection + schema
│   └── research/                # temporal OCR fusion, quality, degradation, evaluation
├── assets/                      # sample or local media
├── path/                        # model files expected by the code
├── requirementss.txt            # root dependency list
└── README.md
```

## Requirements

- Windows was clearly the primary target during development.
- Python 3.10 to 3.12 is the safest choice. Some packages may be rough on 3.13.
- A working camera, RTSP stream, or video file.
- YOLO model weights available locally or downloadable by Ultralytics.
- License plate detector weights at the path configured by `LP_MODEL_PATH`.

## Quick Start

### 1. Create a virtual environment

From the repo root:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirementss.txt
```

Notes:

- The repo currently uses `requirementss.txt` at the root, not `requirements.txt`.
- `torch`, `ultralytics`, `opencv-python`, and OCR packages are already included there.

### 2. Create `.env`

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`.

Minimum useful values:

```env
VIDEO_PATH=../assets/your_video.mp4
LP_MODEL_PATH=../path/to/LP-detection.pt
TARGET_WIDTH=2560
TARGET_HEIGHT=1440
MAX_CAPACITY=80
```

Optional database:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname?sslmode=require
```

If `DATABASE_URL` is missing or invalid, the core pipeline should still run with local file outputs.

### 3. Make sure model files exist

The code expects:

- vehicle detector model through Ultralytics YOLO
- license plate detector at `LP_MODEL_PATH`
- optional Real-ESRGAN weights for plate upscaling

Important:

- `PKOD1/tracking/detector.py` defaults to `path/to/yolo11m.pt`, and falls back to `yolov8x.pt` if it cannot load that file.
- `PKOD1/ocr_processor.py` expects the plate detector at `LP_MODEL_PATH`.

If your local model paths are different, update `.env` or `PKOD1/config.py`.

## How To Run The Project

Open separate terminals inside the repo.

### Terminal 1: run the main live pipeline

```powershell
. .venv\Scripts\Activate.ps1
cd PKOD1
python main.py
```

This starts:

- video capture
- vehicle detection and tracking
- occupancy counting
- ROI observation
- OCR job creation
- live overlays

Keyboard controls in the main window:

- `q` = quit
- `r` = reset occupancy and runtime state
- `s` = manually set occupancy from console input

### Terminal 2: run the OCR processor

```powershell
. .venv\Scripts\Activate.ps1
cd PKOD1
python ocr_processor.py
```

This worker:

- watches `ocr_jobs/`
- processes pending saved frame sets
- detects license plates
- runs OCR
- applies temporal fusion modes if enabled
- writes local OCR outputs
- optionally logs to PostgreSQL

### Terminal 3: optional admin dashboard

```powershell
. .venv\Scripts\Activate.ps1
cd PKOD1
python admin_dashboard.py
```

Use this if you want:

- reset system
- force full
- set occupancy manually
- inspect current snapshot state

## Current Runtime Outputs

Common generated outputs include:

- `PKOD1/ocr_jobs/` for pending and processed OCR jobs
- `PKOD1/plate_crops/` for saved plate crops
- `PKOD1/ocr_results.jsonl` for local OCR result records
- `PKOD1/ocr_results.json` for the latest per-track OCR snapshot
- `PKOD1/ocr_evaluation/` for research evaluation exports
- `occupancy_state.json.tmp` for local parking state snapshot

## Temporal OCR Research Layer

The repo now includes an optional research extension for video OCR fusion.

New research features:

- rolling multi-frame OCR buffering
- frame quality scoring
- temporal OCR fusion
- synthetic image degradation for benchmarking
- evaluation metrics export
- OCR overlays showing current and fused text

### Experiment Modes

Set `OCR_EXPERIMENT_MODE` in `.env` or `PKOD1/config.py`:

- `A` = single-frame OCR baseline
- `B` = temporal majority voting
- `C` = confidence-weighted temporal fusion
- `D` = quality-weighted character fusion

### Important research toggles

Key settings live in [PKOD1/config.py](/abs/path/c:/Users/Adarsh/OneDrive/Documents/PKOD/PKOD1/config.py:71):

- `ENABLE_TEMPORAL_OCR`
- `TEMPORAL_BUFFER_SIZE`
- `TEMPORAL_BUFFER_CLEANUP_SECS`
- `TEMPORAL_WEIGHTING_STRATEGY`
- `OCR_EXPERIMENT_MODE`
- `OCR_EVALUATION_ENABLED`
- `OCR_GROUND_TRUTH_FILE`
- `OCR_SYNTHETIC_DEGRADATIONS`

### Example research config

```env
ENABLE_TEMPORAL_OCR=1
OCR_EXPERIMENT_MODE=D
TEMPORAL_BUFFER_SIZE=12
TEMPORAL_BUFFER_CLEANUP_SECS=2.5
OCR_EVALUATION_ENABLED=1
OCR_GROUND_TRUTH_FILE=ground_truth.json
OCR_SYNTHETIC_DEGRADATIONS=["motion_blur","jpeg_compression"]
```

Note:

- `OCR_SYNTHETIC_DEGRADATIONS` is parsed as JSON first.
- If JSON parsing fails, it falls back to comma-separated values.

## Running Evaluation

If you have ground-truth labels, create a JSON file such as:

```json
{
  "track_12_1712345678901": "MH12AB1234",
  "12": "MH12AB1234"
}
```

Then run the system normally with:

- `OCR_EVALUATION_ENABLED=1`
- `OCR_GROUND_TRUTH_FILE=your_ground_truth.json`

After some OCR jobs are processed, export or refresh summaries with:

```powershell
. .venv\Scripts\Activate.ps1
cd PKOD1
python research_runner.py
```

Expected evaluation files:

- `PKOD1/ocr_evaluation/records.jsonl`
- `PKOD1/ocr_evaluation/summary.json`
- `PKOD1/ocr_evaluation/summary.csv`

Metrics currently include:

- character accuracy
- full plate accuracy
- edit distance
- OCR confidence mean
- temporal improvement percentage over the single-frame baseline

## How To Work On This Repo

This repo already has a working live pipeline. The safest way to contribute is:

1. Keep the detector and OCR internals intact unless you are fixing a clear bug.
2. Add new features as modules around the existing flow.
3. Keep `main.py` as the live orchestrator and `ocr_processor.py` as the async OCR worker.
4. Prefer config flags over hard behavior changes.
5. For safer debugging, test on a short recorded clip first, then move to CCTV/RTSP/live feeds once the ROI and model paths are correct.

Good extension points:

- `PKOD1/research/` for benchmarking and fusion work
- `PKOD1/ui/overlay.py` for extra visualization
- `PKOD1/roi_ocr/` for ROI-trigger logic
- `PKOD1/db/` for result persistence

## Troubleshooting

### `python main.py` fails because models are missing

Check:

- `VIDEO_PATH`
- `LP_MODEL_PATH`
- YOLO weight path or Ultralytics fallback behavior

### OCR worker starts but never finds jobs

Make sure:

- `python main.py` is running
- the vehicle actually enters the configured `PLATE_ROI`
- `ocr_jobs/` is being created under `PKOD1/`

### Database errors appear on startup

If the DB is optional for your use case, remove `DATABASE_URL` from `.env` and rely on local files first.

### Results look wrong because of ROI placement

The plate ROI is currently configured directly in `PKOD1/config.py`:

- `PLATE_ROI`
- `LINE`
- resolution settings like `TARGET_WIDTH` and `TARGET_HEIGHT`

These values must match your camera/video framing.

### OCR research output is empty

Check:

- `ENABLE_TEMPORAL_OCR=1`
- `OCR_EXPERIMENT_MODE`
- whether enough frames are buffered
- whether the OCR worker is running in parallel with `main.py`

## Useful Commands

Install dependencies:

```powershell
pip install -r requirementss.txt
```

Run main pipeline:

```powershell
cd PKOD1
python main.py
```

Run OCR worker:

```powershell
cd PKOD1
python ocr_processor.py
```

Run admin dashboard:

```powershell
cd PKOD1
python admin_dashboard.py
```

Run evaluation summary:

```powershell
cd PKOD1
python research_runner.py
```

Clean OCR artifacts:

```powershell
cd PKOD1
python cleanup_ocr.py
```
