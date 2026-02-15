import cv2

# =========================
# CONFIG — MUST MATCH YOUR PIPELINE
# =========================
VIDEO_SOURCE = "assets/D61_S20251201080035_E20251201080639.mp4"   # or 0 for webcam
DISPLAY_W = 1280            # same as your processing width
DISPLAY_H = 720             # same as your processing height

# =========================
# STATE
# =========================
roi_start = None
roi_end = None
drawing = False

# =========================
# MOUSE CALLBACK
# =========================
def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        drawing = False

        x1, y1 = roi_start
        x2, y2 = roi_end

        # normalize (top-left → bottom-right)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        print(f"\nROI = x1={x1}, y1={y1}, x2={x2}, y2={y2}")

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("❌ Failed to open video source")
    exit(1)

cv2.namedWindow("ROI_CALIBRATION", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("ROI_CALIBRATION", mouse_callback)

print("🟢 Draw ROI with mouse")
print("🟢 Press ESC to exit")

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 🔽 RESIZE TO MATCH PIPELINE
    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

    if roi_start and roi_end:
        cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)

    cv2.imshow("ROI_CALIBRATION", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
