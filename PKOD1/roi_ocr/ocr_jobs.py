import os
import time
import json
import cv2 as cv

OCR_JOB_DIR = "ocr_jobs"

# Supabase integration (fail-safe)
try:
    from db import repository as db_repo
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    db_repo = None

def fire_ocr_job(vs, debug=False):
    """Create an OCR job directory from `vs.ocr_frame_buffer`.

    This function must not perform OCR; it only writes frames and a job.json
    and returns immediately. It is safe to call multiple times, but callers
    should check `vs.ocr_fired` to avoid duplicates.
    """
    if vs is None:
        return
    # safety: ensure we don't create empty jobs
    if len(getattr(vs, 'ocr_frame_buffer', ())) == 0:
        if debug:
            print(f"[OCR JOB SKIPPED] No buffered frames for ID {vs.id}")
        return

    os.makedirs(OCR_JOB_DIR, exist_ok=True)
    job_id = f"{vs.id}_{int(time.time())}"
    job_path = os.path.join(OCR_JOB_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    frame_files = []
    for i, f in enumerate(vs.ocr_frame_buffer):
        fname = f"frame_{i}.jpg"
        fpath = os.path.join(job_path, fname)
        try:
            cv.imwrite(fpath, f)
            frame_files.append(fname)
        except Exception:
            pass

    job_meta = {
        "job_id": job_id,
        "vehicle_id": vs.id,
        "event": "ENTRY" if vs.has_entered else "EXIT",
        "timestamp": time.time(),
        "frames": frame_files,
        "status": "pending",
    }

    try:
        with open(os.path.join(job_path, "job.json"), "w", encoding="utf-8") as jf:
            json.dump(job_meta, jf, indent=2)
    except Exception:
        pass

    # mark fired to prevent duplicates at caller level
    vs.ocr_fired = True

    # Log OCR job to Supabase (plate_text will be null until OCR runs)
    if _DB_AVAILABLE:
        event = "entry" if vs.has_entered else "exit"
        first_frame_path = os.path.join(job_path, frame_files[0]) if frame_files else None
        db_repo.log_ocr_result(
            track_id=vs.id,
            plate_text=None,  # filled later by OCR processor
            confidence=None,
            event_type=event,
            image_path=first_frame_path,
        )

    if debug:
        print(f"[OCR JOB CREATED] {job_id}")
