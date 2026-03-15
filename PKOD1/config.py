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

# --- Base resolution for authored coordinates (4K) ---
_BASE_W, _BASE_H = 3840, 2160
_SCALE_X = TARGET_WIDTH / _BASE_W
_SCALE_Y = TARGET_HEIGHT / _BASE_H

# The Counting Line [x1, y1, x2, y2] — authored at 3840x2160, auto-scaled
_LINE_BASE = [750, 1110, 3300, 1110]
LINE = [int(_LINE_BASE[0] * _SCALE_X), int(_LINE_BASE[1] * _SCALE_Y),
        int(_LINE_BASE[2] * _SCALE_X), int(_LINE_BASE[3] * _SCALE_Y)]

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
MIN_DISPLACEMENT = int(30 * _SCALE_Y)
PRE_ZONE_PIXELS = int(80 * _SCALE_Y)
DIRECTION_STABLE_FRAMES = 4
WARMUP_SECS = 6.0

# State persistence
OCCUPANCY_STATE_FILE = "occupancy_state.json.tmp"
COMMAND_FILE = "admin_commands.json"

OCCUPANCY_AUDIT_LIMIT = 200
LOST_BUFFER_SECS = 2.0
MAX_REASSOC_DISTANCE = int(200 * _SCALE_X)

APPEARANCE_WEIGHT = 0.2

# --- PLATE ROI: Single gate, single ROI (plate visibility window) — authored at 3840x2160, auto-scaled
_ROI_BASE = {"x1": 2124, "y1": 1188, "x2": 3213, "y2": 1824}
PLATE_ROI = {
    "x1": int(_ROI_BASE["x1"] * _SCALE_X),
    "y1": int(_ROI_BASE["y1"] * _SCALE_Y),
    "x2": int(_ROI_BASE["x2"] * _SCALE_X),
    "y2": int(_ROI_BASE["y2"] * _SCALE_Y),
}
MIN_STABLE_FRAMES = 0

# License plate detection model (used by ocr_processor.py)
LP_MODEL_PATH = os.getenv('LP_MODEL_PATH', 'path/to/LP-detection.pt')

# OCR processor polling interval (seconds)
OCR_POLL_INTERVAL = float(os.getenv('OCR_POLL_INTERVAL', '2.0'))
OCR_JOB_DIR = "ocr_jobs"
PLATE_CROPS_DIR = "plate_crops"

# Debug mode
DEBUG = True