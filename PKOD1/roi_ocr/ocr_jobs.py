"""
OCR Job Producer — saves cropped vehicle frames to ocr_jobs/ for the
separate ocr_processor.py to pick up and process.

No OCR dependency here. This module only saves images and metadata.
"""

import os
import time
import json
import cv2 as cv

OCR_JOB_DIR = "ocr_jobs"

# PostgreSQL integration (fail-safe)
try:
    from db import repository as db_repo
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    db_repo = None


def fire_ocr_job(vs, debug=False):
    """Create an OCR job directory from `vs.ocr_frame_buffer`.

    Writes frames as images + metadata.json into a timestamped folder
    under OCR_JOB_DIR. The separate ocr_processor.py watches this
    directory and processes new jobs.
    """
    if not hasattr(vs, 'ocr_frame_buffer') or not vs.ocr_frame_buffer:
        if debug:
            print(f"[OCR JOB] No frames buffered for ID:{vs.id}")
        return

    os.makedirs(OCR_JOB_DIR, exist_ok=True)

    job_id = f"track_{vs.id}_{int(time.time()*1000)}"
    job_path = os.path.join(OCR_JOB_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    # Save buffered frames as images
    frame_files = []
    for i, frame in enumerate(vs.ocr_frame_buffer):
        if frame is None:
            continue
        fname = f"frame_{i:03d}.jpg"
        fpath = os.path.join(job_path, fname)
        try:
            cv.imwrite(fpath, frame)
            frame_files.append(fname)
        except Exception as e:
            if debug:
                print(f"[OCR JOB] Failed to save frame {i}: {e}")

    if not frame_files:
        if debug:
            print(f"[OCR JOB] No valid frames saved for ID:{vs.id}")
        return

    # Write metadata
    event_type = "entry" if vs.has_entered else ("exit" if vs.has_exited else "unknown")
    metadata = {
        "track_id": int(vs.id),
        "event_type": event_type,
        "timestamp": time.time(),
        "frame_count": len(frame_files),
        "frames": frame_files,
        "status": "pending",
    }

    meta_path = os.path.join(job_path, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Mark fired to prevent duplicates
    vs.ocr_fired = True

    # Log OCR job to Supabase (plate_text null until processor fills it)
    if _DB_AVAILABLE:
        first_frame_path = os.path.join(job_path, frame_files[0]) if frame_files else None
        db_repo.log_ocr_result(
            track_id=int(vs.id),
            plate_text=None,
            confidence=None,
            event_type=event_type,
            image_path=first_frame_path,
        )

    if debug:
        print(f"[OCR JOB CREATED] {job_id} — {len(frame_files)} frames")
