# --- CONFIGURATION ---
VIDEO_PATH = "http://10.9.54.207:8080/video"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MAX_CAPACITY = 80

# The Counting Line [x1, y1, x2, y2]
LINE = [250, 400, 1100, 400]

# TRACKING CONFIGURATION
TRACKER_TYPE = "bytetrack"  # Options: "bytetrack" or "botsort"
USE_REID = True  # Enable ReID feature matching for better stability

# Tracker Hyperparameters (Tuned for Parking Lots)
TRACK_HIGH_THRESH = 0.5      # Detection confidence for tracking
TRACK_LOW_THRESH = 0.1       # Low threshold for keeping tracks
NEW_TRACK_THRESH = 0.65      # Higher = harder to create new IDs (reduces ID fragmentation)
TRACK_BUFFER = 90            # Keep lost tracks for 90 frames (~3 sec at 30fps)
MATCH_THRESH = 0.85          # IoU threshold (higher = stricter matching)
MIN_CONFIDENCE = 0.4         # Minimum detection confidence

# ReID Configuration
REID_HISTORY_SIZE = 30       # Store last N frames for feature comparison
REID_SIMILARITY_THRESH = 0.7 # Cosine similarity threshold for ReID matching

# Counting robustness parameters
PRE_QUALIFY_FRAMES = 3      # Frames on approach side required to 'arm' the vehicle
CONFIRM_FRAMES = 2          # Frames beyond the line required to confirm crossing
MIN_DISPLACEMENT = 30       # Minimum pixels beyond the line to accept crossing
COOLDOWN_FRAMES = 30        # Frames to block re-count for same vehicle after event
PRE_ZONE_PIXELS = 80        # Vertical zone (pixels) before/after line for qualification
OCCUPANCY_STATE_FILE = 'occupancy_state.json'
DETECTED_MIN_FRAMES = 5     # Frames required to trust a detection
ARM_MIN_FRAMES = 3          # Frames required to be considered armed near the line
DIRECTION_STABLE_FRAMES = 4 # Frames over which motion direction should be consistent
WARMUP_SECS = 6.0           # Seconds to ignore events after startup
OCCUPANCY_AUDIT_LIMIT = 200 # Max audit entries kept in file
LOST_BUFFER_SECS = 2.0      # Time to keep lost tracklets for re-association (seconds)
MAX_REASSOC_DISTANCE = 200  # Max pixel distance to consider re-association
SPEED_TOLERANCE = 2.0       # Pixels/frame tolerance when comparing speeds
APPEARANCE_WEIGHT = 0.2     # Weight to give appearance in final score (0-1)