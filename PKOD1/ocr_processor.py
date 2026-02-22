"""
Standalone OCR Processor — runs as a SEPARATE process alongside main.py.

Usage:
    cd PKOD1
    python ocr_processor.py

Workflow:
    1. Watches ocr_jobs/ for new job directories (created by main.py)
    2. Loads cropped vehicle frames from each job
    3. Runs YOLO LP-detection.pt to detect license plates
    4. Runs RapidOCR on detected plate crops to extract text
    5. Validates against Indian plate format
    6. Updates Supabase ocr_results with plate_text + confidence
    7. Moves processed jobs to ocr_jobs/processed/
"""

import os
import sys
import time
import json
import re
import shutil
import cv2 as cv
import numpy as np

# Ensure PKOD1 is on the path so config and db modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Supabase integration (fail-safe)
try:
    from db import repository as db_repo
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    db_repo = None
    print("[OCR-PROC] Supabase unavailable — results will be saved locally only")


# ── License plate detection model ──────────────────────────────────

_lp_model = None

def _get_lp_model():
    """Lazy-load the YOLO license plate detection model."""
    global _lp_model
    if _lp_model is not None:
        return _lp_model

    model_path = getattr(config, 'LP_MODEL_PATH', '../path/to/LP-detection.pt')
    abs_path = os.path.abspath(model_path)

    if not os.path.exists(abs_path):
        print(f"[OCR-PROC] ERROR: LP model not found at {abs_path}")
        return None

    try:
        from ultralytics import YOLO
        _lp_model = YOLO(abs_path)
        print(f"[OCR-PROC] LP detection model loaded: {abs_path}")
        return _lp_model
    except Exception as e:
        print(f"[OCR-PROC] Failed to load LP model: {e}")
        return None


# ── RapidOCR engine ────────────────────────────────────────────────

_ocr_engine = None

def _get_ocr_engine():
    """Lazy-load RapidOCR engine."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
        print("[OCR-PROC] RapidOCR engine initialized")
        return _ocr_engine
    except ImportError:
        print("[OCR-PROC] ERROR: rapidocr-onnxruntime not installed")
        print("  Install with: pip install rapidocr-onnxruntime")
        return None
    except Exception as e:
        print(f"[OCR-PROC] Failed to init RapidOCR: {e}")
        return None


# ── Indian license plate validation ────────────────────────────────

# Common Indian plate patterns:
#   KA01AB1234, TN09BY5361, MH12DE1433, DL3CAB1234, etc.
_INDIAN_PLATE_RE = re.compile(
    r'^[A-Z]{2}\s*\d{1,2}\s*[A-Z]{0,3}\s*\d{1,4}$',
    re.IGNORECASE
)

def _clean_plate_text(text: str) -> str:
    """Clean and normalize OCR output for plate validation."""
    if not text:
        return ""
    # Remove common OCR noise characters
    cleaned = text.strip().upper()
    cleaned = re.sub(r'[^A-Z0-9\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _is_valid_plate(text: str) -> bool:
    """Check if text matches an Indian license plate pattern."""
    if not text or len(text) < 4:
        return False
    return bool(_INDIAN_PLATE_RE.match(text))


# ── Core processing ───────────────────────────────────────────────

def _detect_plates(frame):
    """Run YOLO LP-detection on a frame.

    Returns list of (x1, y1, x2, y2, confidence) tuples.
    """
    model = _get_lp_model()
    if model is None:
        return []

    try:
        results = model(frame, verbose=False)
        plates = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                plates.append((x1, y1, x2, y2, conf))
        return plates
    except Exception as e:
        print(f"[OCR-PROC] LP detection error: {e}")
        return []


def _read_plate_text(plate_crop):
    """Run RapidOCR on a plate crop image.

    Returns (text, confidence) or (None, 0.0).
    """
    engine = _get_ocr_engine()
    if engine is None:
        return None, 0.0

    try:
        result, _ = engine(plate_crop)
        if not result:
            return None, 0.0

        # RapidOCR returns list of [box, text, score]
        # Combine all detected text lines
        texts = []
        scores = []
        for item in result:
            if len(item) >= 3:
                texts.append(str(item[1]))
                scores.append(float(item[2]))

        if not texts:
            return None, 0.0

        combined_text = ' '.join(texts)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return combined_text, avg_score

    except Exception as e:
        print(f"[OCR-PROC] RapidOCR error: {e}")
        return None, 0.0


def process_job(job_path: str) -> dict:
    """Process a single OCR job directory.

    Returns dict with: track_id, plate_text, confidence, event_type, valid
    """
    meta_path = os.path.join(job_path, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"[OCR-PROC] No metadata.json in {job_path}")
        return None

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    track_id = metadata.get('track_id', -1)
    event_type = metadata.get('event_type', 'unknown')
    frame_files = metadata.get('frames', [])

    if not frame_files:
        print(f"[OCR-PROC] No frames in job {job_path}")
        return None

    best_plate = None
    best_conf = 0.0
    best_valid = False

    # Process each frame, keep the best plate reading
    for fname in frame_files:
        fpath = os.path.join(job_path, fname)
        if not os.path.exists(fpath):
            continue

        frame = cv.imread(fpath)
        if frame is None:
            continue

        # Step 1: Detect license plates in the frame
        plates = _detect_plates(frame)

        for (x1, y1, x2, y2, det_conf) in plates:
            # Ensure valid crop dimensions
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 - x1 < 10 or y2 - y1 < 5:
                continue

            plate_crop = frame[y1:y2, x1:x2]

            # Step 2: OCR on the plate crop
            text, ocr_conf = _read_plate_text(plate_crop)
            if not text:
                continue

            cleaned = _clean_plate_text(text)
            is_valid = _is_valid_plate(cleaned)
            combined_conf = det_conf * ocr_conf

            # Prefer valid plates; otherwise take highest confidence
            if is_valid and not best_valid:
                best_plate = cleaned
                best_conf = combined_conf
                best_valid = True
            elif (is_valid == best_valid) and combined_conf > best_conf:
                best_plate = cleaned
                best_conf = combined_conf
                best_valid = is_valid

    result = {
        'track_id': track_id,
        'plate_text': best_plate,
        'confidence': round(best_conf, 4),
        'event_type': event_type,
        'valid': best_valid,
    }

    # Update metadata with result
    metadata['status'] = 'processed'
    metadata['result'] = result
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return result


def _update_supabase(result):
    """Update Supabase ocr_results with the processed plate text."""
    if not _DB_AVAILABLE or result is None:
        return

    if result.get('plate_text'):
        db_repo.log_ocr_result(
            track_id=result['track_id'],
            plate_text=result['plate_text'],
            confidence=result['confidence'],
            event_type=result.get('event_type'),
        )


# ── Local results file (backup) ──────────────────────────────────

RESULTS_FILE = "ocr_results.jsonl"

def _save_local_result(result):
    """Append result to local JSONL file as backup."""
    if result is None:
        return
    try:
        with open(RESULTS_FILE, 'a') as f:
            f.write(json.dumps(result) + '\n')
    except Exception as e:
        print(f"[OCR-PROC] Failed to save local result: {e}")


# ── Main loop ────────────────────────────────────────────────────

def run_processor():
    """Main polling loop: watch ocr_jobs/ for new jobs and process them."""
    job_dir = getattr(config, 'OCR_JOB_DIR', 'ocr_jobs')
    poll_interval = getattr(config, 'OCR_POLL_INTERVAL', 2.0)
    processed_dir = os.path.join(job_dir, 'processed')

    print(f"[OCR-PROC] Watching '{job_dir}/' for new jobs (poll every {poll_interval}s)")
    print(f"[OCR-PROC] LP model: {getattr(config, 'LP_MODEL_PATH', 'not configured')}")
    print(f"[OCR-PROC] Supabase: {'connected' if _DB_AVAILABLE else 'disabled'}")
    print("=" * 50)

    # Pre-load models on startup
    _get_lp_model()
    _get_ocr_engine()

    while True:
        try:
            if not os.path.exists(job_dir):
                time.sleep(poll_interval)
                continue

            # Find pending job directories (have metadata.json with status=pending)
            pending_jobs = []
            for entry in os.scandir(job_dir):
                if not entry.is_dir() or entry.name == 'processed':
                    continue
                meta = os.path.join(entry.path, 'metadata.json')
                if os.path.exists(meta):
                    try:
                        with open(meta, 'r') as f:
                            data = json.load(f)
                        if data.get('status') == 'pending':
                            pending_jobs.append(entry.path)
                    except Exception:
                        pass

            if not pending_jobs:
                time.sleep(poll_interval)
                continue

            print(f"[OCR-PROC] Found {len(pending_jobs)} pending job(s)")

            for job_path in pending_jobs:
                job_name = os.path.basename(job_path)
                print(f"[OCR-PROC] Processing: {job_name}")

                result = process_job(job_path)

                if result:
                    status = "✓" if result.get('plate_text') else "✗ no plate"
                    valid = " (valid)" if result.get('valid') else " (invalid format)" if result.get('plate_text') else ""
                    print(f"  → [{status}] ID:{result['track_id']} "
                          f"plate=\"{result.get('plate_text', 'N/A')}\" "
                          f"conf={result.get('confidence', 0):.2f}{valid}")

                    # Save to Supabase
                    _update_supabase(result)

                    # Save local backup
                    _save_local_result(result)

                # Move to processed directory
                os.makedirs(processed_dir, exist_ok=True)
                dest = os.path.join(processed_dir, job_name)
                try:
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.move(job_path, dest)
                except Exception as e:
                    print(f"[OCR-PROC] Failed to move job: {e}")

        except KeyboardInterrupt:
            print("\n[OCR-PROC] Shutting down...")
            break
        except Exception as e:
            print(f"[OCR-PROC] Error in main loop: {e}")
            time.sleep(poll_interval)


if __name__ == '__main__':
    run_processor()
