"""
cleanup_ocr.py
--------------
Deletes all OCR job image folders inside `ocr_jobs/` and
removes any OCR result JSON/JSONL files in the PKOD1 directory.

Run from the PKOD1 directory:
    python cleanup_ocr.py
"""

import os
import shutil
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Delete all content inside ocr_jobs/ ---
OCR_JOBS_DIR = os.path.join(BASE_DIR, "ocr_jobs")

if os.path.exists(OCR_JOBS_DIR):
    deleted_folders = 0
    for item in os.listdir(OCR_JOBS_DIR):
        item_path = os.path.join(OCR_JOBS_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"[DELETED] Folder: {item_path}")
            deleted_folders += 1
        elif os.path.isfile(item_path):
            os.remove(item_path)
            print(f"[DELETED] File: {item_path}")
    if deleted_folders == 0:
        print("[INFO] ocr_jobs/ is already empty.")
else:
    print(f"[INFO] ocr_jobs/ directory not found at {OCR_JOBS_DIR}, skipping.")

# --- 2. Delete all content inside plate_crops/ ---
PLATE_CROPS_DIR = os.path.join(BASE_DIR, "plate_crops")

if os.path.exists(PLATE_CROPS_DIR):
    deleted_pc = 0
    for item in os.listdir(PLATE_CROPS_DIR):
        item_path = os.path.join(PLATE_CROPS_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"[DELETED] Folder: {item_path}")
            deleted_pc += 1
        elif os.path.isfile(item_path):
            os.remove(item_path)
            print(f"[DELETED] File: {item_path}")
    if deleted_pc == 0:
        print("[INFO] plate_crops/ is already empty.")
else:
    print(f"[INFO] plate_crops/ directory not found, skipping.")

# --- 3. Delete OCR result JSON/JSONL files ---
JSON_PATTERNS = [
    os.path.join(BASE_DIR, "ocr_results.json"),
    os.path.join(BASE_DIR, "ocr_results.jsonl"),
]

deleted_json = 0
for pattern in JSON_PATTERNS:
    for f in glob.glob(pattern):
        os.remove(f)
        print(f"[DELETED] {f}")
        deleted_json += 1

if deleted_json == 0:
    print("[INFO] No OCR result JSON files found to delete.")

print("\n✅ Cleanup complete.")
