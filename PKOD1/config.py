# --- CONFIGURATION ---
# Loads settings from .env when available, with sensible defaults.

import os
try:
    from dotenv import load_dotenv
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    load_dotenv(os.path.join(_root, '.env'))
except ImportError:
    pass


# Video source (file path, camera index, or RTSP URL)
VIDEO_PATH = os.getenv('VIDEO_PATH', '../source/D61_S20251201080035_E20251201080639.mp4')
TARGET_WIDTH = int(os.getenv('TARGET_WIDTH', '3840'))
TARGET_HEIGHT = int(os.getenv('TARGET_HEIGHT', '2160'))

MAX_CAPACITY = int(os.getenv('MAX_CAPACITY', '80'))

# The Counting Line [x1, y1, x2, y2] — scaled for 3840x2160
LINE = [750, 1110, 3300, 1110]

# TRACKING CONFIGURATION
TRACKER_TYPE = "bytetrack"
USE_REID = True

TRACK_HIGH_THRESH = 0.5
TRACK_LOW_THRESH = 0.1
NEW_TRACK_THRESH = 0.65
TRACK_BUFFER = 90
MATCH_THRESH = 0.85
MIN_CONFIDENCE = 0.4

# ReID
REID_HISTORY_SIZE = 30
REID_SIMILARITY_THRESH = 0.7

# Counting robustness
DETECTED_MIN_FRAMES = 5
ARM_MIN_FRAMES = 3
CONFIRM_FRAMES = 2
MIN_DISPLACEMENT = 30
PRE_ZONE_PIXELS = 80
DIRECTION_STABLE_FRAMES = 4
WARMUP_SECS = 6.0

# State persistence
OCCUPANCY_STATE_FILE = "occupancy_state.json.tmp"
COMMAND_FILE = "admin_commands.json"

OCCUPANCY_AUDIT_LIMIT = 200
LOST_BUFFER_SECS = 2.0
MAX_REASSOC_DISTANCE = 200

APPEARANCE_WEIGHT = 0.2

# --- PLATE ROI: Single gate, single ROI (plate visibility window) — scaled for 3840x2160
PLATE_ROI = {"x1": 2124, "y1": 1188, "x2": 3213, "y2": 1824}
MIN_STABLE_FRAMES = 0

# License plate detection model (used by ocr_processor.py)
LP_MODEL_PATH = os.getenv('LP_MODEL_PATH', 'path/to/LP-detection.pt')

# OCR processor polling interval (seconds)
OCR_POLL_INTERVAL = float(os.getenv('OCR_POLL_INTERVAL', '2.0'))
OCR_JOB_DIR = "ocr_jobs"
PLATE_CROPS_DIR = "plate_crops"

# Debug mode
DEBUG = True