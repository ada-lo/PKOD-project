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
import urllib.request
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
PLATE_CROPS_DIR = getattr(config, 'PLATE_CROPS_DIR', 'plate_crops')

# ── AI Super-Resolution (Real-ESRGAN x4fast) ─────────────────────────────────
# Uses SRVGGNetCompact (compact/fast variant). Weights auto-download on first use (~17MB).
# Falls back to Lanczos4 + unsharp masking if unavailable.

_REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
_REALESRGAN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "path", "to", "realesr-animevideov3.pth")
_sr_model = None
_sr_model_tried = False


def _get_sr_model():
    """Lazy-load Real-ESRGAN x4fast model. Auto-downloads weights if missing."""
    global _sr_model, _sr_model_tried
    if _sr_model_tried:
        return _sr_model
    _sr_model_tried = True

    # Auto-download weights if not present
    if not os.path.exists(_REALESRGAN_MODEL_PATH):
        try:
            print(f"[SR] Downloading Real-ESRGAN x4fast weights (~17MB) to {_REALESRGAN_MODEL_PATH} ...")
            os.makedirs(os.path.dirname(_REALESRGAN_MODEL_PATH), exist_ok=True)
            urllib.request.urlretrieve(_REALESRGAN_URL, _REALESRGAN_MODEL_PATH)
            print("[SR] Download complete.")
        except Exception as e:
            print(f"[SR] Could not download Real-ESRGAN weights: {e} — using fallback upscaling")
            return None

    try:
        import torch
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        from realesrgan import RealESRGANer

        arch = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_conv=16,
            upscale=4, act_type='prelu'
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        upsampler = RealESRGANer(
            scale=4,
            model_path=_REALESRGAN_MODEL_PATH,
            model=arch,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),  # fp16 on GPU for speed
            device=device,
        )
        _sr_model = upsampler
        print(f"[SR] Real-ESRGAN x4fast loaded on {device}.")
        return _sr_model
    except Exception as e:
        print(f"[SR] Failed to load Real-ESRGAN: {e} — using fallback upscaling")
        return None


def _upscale_plate(img):
    """Upscale a plate crop image.

    Primary:  Real-ESRGAN x4fast (SRVGGNetCompact).
    Fallback: Lanczos4 x4 interpolation + unsharp masking.

    Always returns a BGR image at 4x the input resolution.
    """
    sr = _get_sr_model()
    if sr is not None:
        try:
            # RealESRGANer expects BGR uint8 input, returns BGR uint8
            output, _ = sr.enhance(img, outscale=4)
            return output
        except Exception as e:
            print(f"[SR] Real-ESRGAN inference failed: {e} — falling back to Lanczos")

    # Fallback: Lanczos4 + unsharp masking
    h, w = img.shape[:2]
    upscaled = cv.resize(img, (w * 4, h * 4), interpolation=cv.INTER_LANCZOS4)
    blurred = cv.GaussianBlur(upscaled, (0, 0), 3)
    sharpened = cv.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    return sharpened


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


# ── Plate crop saver ─────────────────────────────────────────────

def _save_plate_crop(job_id, track_id, event_type, frame_index, plate_index, plate_crop, det_conf):
    """Save a single detected plate crop + upscaled version + metadata to plate_crops/<job_id>/."""
    try:
        job_dir = os.path.join(PLATE_CROPS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        img_name = f"plate_{frame_index:03d}_{plate_index}.jpg"
        img_path = os.path.join(job_dir, img_name)
        cv.imwrite(img_path, plate_crop)

        # Upscale and save alongside original
        upscaled_name = f"plate_{frame_index:03d}_{plate_index}_upscaled.jpg"
        upscaled_path = os.path.join(job_dir, upscaled_name)
        try:
            upscaled = _upscale_plate(plate_crop)
            cv.imwrite(upscaled_path, upscaled)
        except Exception as e:
            print(f"[OCR-PROC] Upscaling failed for {img_name}: {e}")
            upscaled_name = None

        # Load or create the shared metadata.json for this job
        meta_path = os.path.join(job_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "track_id": int(track_id),
                "event_type": event_type,
                "timestamp": time.time(),
                "plates": [],
            }

        metadata["plates"].append({
            "filename": img_name,
            "upscaled_filename": upscaled_name,
            "detection_confidence": round(float(det_conf), 4),
            "frame_index": frame_index,
            "status": "pending_ocr",
        })

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"[OCR-PROC] Failed to save plate crop: {e}")


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
    job_id = os.path.basename(job_path)

    # Process each frame, keep the best plate reading
    for frame_index, fname in enumerate(frame_files):
        fpath = os.path.join(job_path, fname)
        if not os.path.exists(fpath):
            continue

        frame = cv.imread(fpath)
        if frame is None:
            continue

        # Step 1: Detect license plates in the frame
        plates = _detect_plates(frame)

        for plate_index, (x1, y1, x2, y2, det_conf) in enumerate(plates):
            # Ensure valid crop dimensions
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 - x1 < 10 or y2 - y1 < 5:
                continue

            plate_crop = frame[y1:y2, x1:x2]

            # Save the plate crop to plate_crops/ folder
            _save_plate_crop(job_id, track_id, event_type,
                             frame_index, plate_index, plate_crop, det_conf)

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
