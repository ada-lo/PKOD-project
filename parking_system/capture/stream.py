import cv2 as cv
import time

def _is_stream(src: str):
    if not isinstance(src, str):
        return False
    s = src.lower()
    return s.startswith('http://') or s.startswith('https://') or s.startswith('rtsp://')

def open_capture(src):
    """Open video capture from file, camera index, or network stream."""
    # Prefer FFMPEG for network streams when available
    if _is_stream(src):
        print(f"Opening network stream: {src}")
        try:
            cap = cv.VideoCapture(src, cv.CAP_FFMPEG)
        except Exception:
            cap = cv.VideoCapture(src)
    else:
        # try numeric index
        try:
            idx = int(src)
            cap = cv.VideoCapture(idx)
        except Exception:
            cap = cv.VideoCapture(src)

    # quick warm-up grabs (do not rely on returned frames here)
    try:
        for _ in range(2):
            cap.grab()
    except Exception:
        pass
    return cap