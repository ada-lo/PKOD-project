import cv2 as cv
import time
import threading
import os
import config
import json
from collections import deque
from capture.stream import open_capture, _is_stream
from tracking.detector import Detector
from identity.vehicle_tracker import VehicleTracker
from identity.tracklet_buffer import TrackletBuffer
from events.event_manager import VehicleEventManager
from state.occupancy_store import load_occupancy, save_occupancy, load_vehicle_states
from ui.overlay import draw_bounding_box, draw_counting_line, draw_ui_overlay, draw_full_message

# ROI/OCR observer — buffers frames for the separate OCR processor
try:
    from roi_ocr.roi_observer import observe_roi
except Exception:
    def observe_roi(vs, frame):
        return

# OCR job producer — saves cropped frames to ocr_jobs/ for the processor
try:
    from roi_ocr.ocr_jobs import fire_ocr_job
except Exception:
    def fire_ocr_job(vs, debug=False):
        return

# PostgreSQL event logging (fail-safe)
try:
    from db.connection import init_db
    from db import repository as db_repo
    _DB_AVAILABLE = True
    # Auto-create tables on startup
    init_db()
except ImportError:
    _DB_AVAILABLE = False
    db_repo = None


# ROI and OCR logic moved to roi_ocr package to strictly separate concerns.


# NOTE: removed unused save_debug_crop_async/_save_debug_crop helpers —
# OCR saving is handled by save_debug_crop_and_ocr_async -> _ocr_worker


DEBUG = getattr(config, 'DEBUG', True)

def check_admin_commands(occupancy, entry_count, exit_count, frozen):
    """
    Safely process admin commands from dashboard.
    Returns: occupancy, entry_count, exit_count, frozen, updated
    """
    if not os.path.exists(config.COMMAND_FILE):
        return occupancy, entry_count, exit_count, frozen, False

    try:
        with open(config.COMMAND_FILE, "r") as f:
            cmd = json.load(f)

        os.remove(config.COMMAND_FILE)

        now = time.time()
        ts = cmd.get("ts", 0)

        # Reject stale commands
        if abs(now - ts) > 10:
            print("[ADMIN] Ignored stale command")
            return occupancy, entry_count, exit_count, frozen, False

        command = cmd.get("command")
        value = cmd.get("value", 0)

        print(f"[ADMIN] Command received: {command}")

        if command == "RESET_SYSTEM":
            return 0, 0, 0, False, True

        if command == "SET_OCCUPANCY":
            new_occ = max(0, min(int(value), config.MAX_CAPACITY))
            return new_occ, entry_count, exit_count, False, True

        if command == "FORCE_FULL":
            return config.MAX_CAPACITY, entry_count, exit_count, True, True

        if command == "FREEZE":
            return occupancy, entry_count, exit_count, True, True

        if command == "RESUME_AUTO":
            return occupancy, entry_count, exit_count, False, True

    except Exception as e:
        print(f"[ADMIN] Command error: {e}")

    return occupancy, entry_count, exit_count, frozen, False

# --- INITIALIZATION ---
print("=== ENHANCED PARKING SYSTEM ===")
print(f"Tracker: {config.TRACKER_TYPE.upper()}")
print(f"ReID Matching: {'ENABLED' if config.USE_REID else 'DISABLED'}")
print(f"Track Buffer: {config.TRACK_BUFFER} frames")
print("=" * 35)

# Validate source
if config.VIDEO_PATH == "":
    print("No VIDEO_PATH set; defaulting to camera 0")
    video_source = 0
else:
    video_source = config.VIDEO_PATH

if isinstance(video_source, str) and not (os.path.exists(video_source) or _is_stream(video_source)):
    print(f"CRITICAL ERROR: File not found at {video_source}")
    exit()

# Initialize components
detector = Detector(min_confidence=config.MIN_CONFIDENCE)
tracklet_buffer = TrackletBuffer()
vehicle_tracker = VehicleTracker(tracklet_buffer=tracklet_buffer)
event_manager = VehicleEventManager()

# Initialize video capture
vid = open_capture(video_source)

# Load persisted occupancy and vehicle states (snapshot recovery)
occupancy, entry_count, exit_count, last_update = load_occupancy()
vehicle_states_list = load_vehicle_states()
# runtime in-memory vehicle state: id -> VehicleState
class VehicleState:
    def __init__(self, id, has_entered=False, has_exited=False, edge_frames=0, ocr_armed=False, ocr_fired=False):
        self.id = id
        self.has_entered = has_entered
        self.has_exited = has_exited

        # edge_frames measures temporal stability of plate visibility.
        self.edge_frames = edge_frames
        # OCR arming indicates stable plate visibility only.
        self.ocr_armed = ocr_armed
        # OCR firing flag: one-time snapshot attempt after event confirmation.
        self.ocr_fired = ocr_fired

        # bounded frame buffer; store only copies
        self.ocr_frame_buffer = deque(maxlen=10)


vehicle_states = {v['id']: VehicleState(v['id'], has_entered=v.get('has_entered', False), has_exited=v.get('has_exited', False)) for v in vehicle_states_list}
# missing frame counters for cleanup
missing_frames = {v['id']: 0 for v in vehicle_states_list}
# max allowed missing frames before considering deletion of exited states
MAX_MISSING = 15

# NOTE: keep `occupancy` loaded from persistent store (do NOT override from vehicle_states)
# This preserves counting behavior identical to PKODD (do not derive occupancy here).
# occupancy remains as returned by load_occupancy()

# Warm-up and freeze state
startup_ts = time.time()
warmup_end = startup_ts + config.WARMUP_SECS
frozen = occupancy < 0 or occupancy > config.MAX_CAPACITY

# Counting state
history = {}
frame_count = 0

print("\nStarting tracking loop. Press 'Q' to exit.\n")

# Main loop
while True:
    success, frame = vid.read()
    if not success or frame is None:
        print("Stream read failed — attempting reconnect in 1s...")
        try:
            vid.release()
        except Exception:
            pass
        time.sleep(1.0)
        vid = open_capture(video_source)
        continue

    frame_count += 1
     
    occupancy, entry_count, exit_count, frozen, updated = check_admin_commands(
        occupancy, entry_count, exit_count, frozen
    )

    if updated:
        save_occupancy(
            occupancy,
            entry_count,
            exit_count,
            reason="admin_command",
            vehicle_states=[vars(v) for v in vehicle_states.values()],
        )
        
    frame = cv.resize(frame, (config.TARGET_WIDTH, config.TARGET_HEIGHT))

    # Draw counting line and UI overlay (overlay draws the ROI)
    draw_counting_line(frame)

    # Run detection
    results = detector.detect(frame)

    current_frame_ids = []
    # collect per-vehicle debug overlays to render after UI overlay (avoid being covered)
    debug_overlays = []
    # collector for Phase 2 observer tasks: run AFTER counting phase completes
    observer_tasks = []
    if results:
        boxes, ids, confs = detector.extract_detections(results)

        if len(boxes) > 0:
            for box, track_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                # ReID: Try to match with lost tracks
                original_id = track_id
                track_id = vehicle_tracker.match_lost_track(frame, track_id, box, (cx, cy))

                # Update tracking history
                vehicle_tracker.update(frame, track_id, box, (cx, cy))
                current_frame_ids.append(track_id)

                # Creation: create vehicle state only in tracker loop
                if track_id not in vehicle_states:
                    vehicle_states[track_id] = VehicleState(id=track_id)
                    missing_frames[track_id] = 0

                # Draw bounding box (capture returned center)
                is_reid = track_id != original_id
                cx, cy = draw_bounding_box(frame, box, track_id, conf, is_reid=is_reid)

                # --- DIRECTION (used by observer only) ---
                # Determine recent motion to infer direction. TOP->BOTTOM (dy>0)=ENTRY
                pos_hist = vehicle_tracker.position_history.get(track_id, [])
                if len(pos_hist) >= 2:
                    mv = pos_hist[-1][1] - pos_hist[0][1]
                else:
                    mv = 0

                direction = 'ENTRY' if mv > 0 else 'EXIT'

                # Defer ROI/OCR observer to Phase 2: collect necessary info now
                observer_tasks.append((track_id, (x1, y1, x2, y2), direction))

                # --- COUNTING LOGIC ---
                if config.LINE[0] < cx < config.LINE[2]:
                    pos_hist = vehicle_tracker.position_history.get(track_id, [])
                    evt = event_manager.process(track_id, cy, pos_hist, frame_count)

                    if evt in ['entry', 'exit']:
                        now_ts = time.time()

                        if now_ts < warmup_end:
                            print(f"[WARMUP] Ignored {evt} for ID:{track_id} (startup warm-up)")
                        elif frozen:
                            print(f"[FROZEN] Ignored {evt} for ID:{track_id} — occupancy frozen. Press 'r' to reset.")
                        else:
                            if evt == 'entry':
                                if 'entry' not in event_manager.counted.get(track_id, set()):
                                    if occupancy < config.MAX_CAPACITY:
                                        event_manager.counted[track_id].add('entry')
                                        entry_count += 1
                                        occupancy += 1
                                        draw_counting_line(frame, highlight_color=(0, 255, 0))
                                        print(f"[ENTRY] ID:{track_id} | Occupancy: {occupancy}/{config.MAX_CAPACITY}")
                                        save_occupancy(occupancy, entry_count, exit_count, reason='entry', vehicle_states=[vars(v) for v in vehicle_states.values()])
                                        # Log to Supabase
                                        if _DB_AVAILABLE:
                                            db_repo.log_vehicle_event(track_id, 'entry', occupancy)
                                    else:
                                        draw_full_message(frame, "FULL - NO ENTRY", (cx - 80, cy - 30))
                                        print(f"[BLOCKED] ID:{track_id} - Parking FULL")

                            else:  # exit
                                if 'exit' not in event_manager.counted.get(track_id, set()):
                                    event_manager.counted[track_id].add('exit')
                                    exit_count += 1
                                    occupancy = max(0, occupancy - 1)
                                    draw_counting_line(frame, highlight_color=(0, 255, 0))
                                    print(f"[EXIT] ID:{track_id} | Occupancy: {occupancy}/{config.MAX_CAPACITY}")
                                    save_occupancy(occupancy, entry_count, exit_count, reason='exit', vehicle_states=[vars(v) for v in vehicle_states.values()])
                                    # Log to Supabase
                                    if _DB_AVAILABLE:
                                        db_repo.log_vehicle_event(track_id, 'exit', occupancy)

                            if occupancy < 0 or occupancy > config.MAX_CAPACITY:
                                print("[ALERT] Occupancy invariant violated — freezing counting until manual reset")
                                frozen = True

                # Store last center for basic history
                history[track_id] = cy

            # Update lost track monitoring
            vehicle_tracker.mark_present_ids(current_frame_ids)

            # Prune buffered lost tracklets
            if tracklet_buffer is not None:
                tracklet_buffer.prune()

            # PHASE 2 — OBSERVER ONLY (must run after counting completed)
            for task in observer_tasks:
                t_id, bbox, direction = task
                x1, y1, x2, y2 = bbox
                # ensure vs exists
                vs = vehicle_states.get(t_id)
                if vs is None:
                    vs = VehicleState(id=t_id)
                    vehicle_states[t_id] = vs

                # Expose direction and bbox to observer only now
                vs.direction = direction
                vs.bbox = (x1, y1, x2, y2)

                # Safety: verify observer cannot alter counts or vehicle flags
                before_counts = (entry_count, exit_count)
                before_flags = (bool(vs.has_entered), bool(vs.has_exited))
                prev_ocr_fired = bool(getattr(vs, 'ocr_fired', False))
                try:
                    observe_roi(vs, frame)
                except Exception:
                    pass
                # Enforce observer did not change counting or lifecycle flags
                assert (entry_count, exit_count) == before_counts, f"Observer modified counts for ID {t_id}"
                assert (bool(vs.has_entered), bool(vs.has_exited)) == before_flags, f"Observer modified lifecycle flags for ID {t_id}"

                # If observer just fired OCR, perform crop here to guarantee execution
                post_ocr_fired = bool(getattr(vs, 'ocr_fired', False))
                if post_ocr_fired and not prev_ocr_fired:
                    try:
                        fh, fw = frame.shape[:2]
                        ix1 = max(0, int(x1))
                        iy1 = max(0, int(y1))
                        ix2 = min(fw, int(x2))
                        iy2 = min(fh, int(y2))
                        if ix2 > ix1 and iy2 > iy1:
                            crop = frame[iy1:iy2, ix1:ix2].copy()
                            # append to buffer and write proof file
                            try:
                                vs.ocr_frame_buffer.append(crop)
                            except Exception:
                                pass
                            try:
                                tmp_dir = os.path.join('ocr_jobs', 'main_tmp')
                                os.makedirs(tmp_dir, exist_ok=True)
                                tmp_name = f"main_crop_{t_id}_{int(time.time())}.jpg"
                                cv.imwrite(os.path.join(tmp_dir, tmp_name), crop)
                            except Exception:
                                pass
                            # Now that the crop exists in the buffer, create the OCR job
                            try:
                                fire_ocr_job(vs, debug=getattr(config, 'DEBUG', False))
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Prepare debug overlay text for rendering after UI overlay
                dbg_text = f"ID {t_id} | dir={direction} | edge_frames={getattr(vs,'edge_frames',0)}"
                if vs and getattr(vs, 'ocr_armed', False):
                    dbg_text += " | OCR_ARMED"
                if vs and getattr(vs, 'ocr_fired', False):
                    dbg_text += " | OCR_FIRED"
                dbg_x = max(0, x1)
                dbg_y = max(20, y1 - 10)
                debug_overlays.append((dbg_text, (dbg_x, dbg_y)))

    # Missing tracker cleanup: update missing_frames and remove exited states after timeout
    active_ids = set(current_frame_ids)
    for tid in list(vehicle_states.keys()):
        if tid not in active_ids:
            missing_frames[tid] = missing_frames.get(tid, 0) + 1
        else:
            missing_frames[tid] = 0

    for tid in list(missing_frames.keys()):
        miss = missing_frames.get(tid, 0)
        vs = vehicle_states.get(tid)
        if miss >= MAX_MISSING:
            # safe to delete only if exited
            if vs and vs.has_exited:
                del vehicle_states[tid]
                del missing_frames[tid]
            else:
                # tracker lost before exit — keep state (prevents ghost inheritance)
                pass

    # Draw UI overlay
    warmup_remaining = max(0, int(warmup_end - time.time()))
    lost_tracks_count = len(vehicle_tracker.lost_tracks)
    draw_ui_overlay(frame, occupancy, entry_count, exit_count, frozen, 
                   warmup_remaining, lost_tracks_count)

    # Render per-vehicle debug overlays on top of UI
    for txt, (dx, dy) in debug_overlays:
        cv.putText(frame, txt, (dx, dy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display frame
    cv.imshow("Enhanced Parking System v2.0", frame)

    # Handle keyboard input
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('r'):
        # Manual reset to empty-start
        occupancy = 0
        entry_count = 0
        exit_count = 0
        frozen = False
        # clear vehicle runtime state as well on manual reset
        vehicle_states = {}
        missing_frames = {}
        save_occupancy(occupancy, entry_count, exit_count, reason='manual_reset', vehicle_states=[])
        # Clear vehicle states in Supabase
        if _DB_AVAILABLE:
            db_repo.clear_vehicle_states()
        print("[MANUAL] Occupancy reset to 0 by operator")
    elif k == ord('s'):
        # Manual set occupancy value via console prompt
        try:
            val = int(input("Set occupancy to (integer): "))
            val = max(0, min(val, config.MAX_CAPACITY))
            occupancy = val
            frozen = False
            save_occupancy(occupancy, entry_count, exit_count, reason='manual_set', vehicle_states=[vars(v) for v in vehicle_states.values()])
            print(f"[MANUAL] Occupancy set to {occupancy} by operator")
        except Exception as e:
            print(f"Invalid input for occupancy set: {e}")

# Cleanup
vid.release()
cv.destroyAllWindows()

# Save final state
save_occupancy(occupancy, entry_count, exit_count, vehicle_states=[vars(v) for v in vehicle_states.values()])

# Cleanup tracker config file
tracker_config = f"{config.TRACKER_TYPE}_custom.yaml"
if os.path.exists(tracker_config):
    os.remove(tracker_config)

print("\n=== SESSION SUMMARY ===")
print(f"Total Entries: {entry_count}")
print(f"Total Exits: {exit_count}")
print(f"Final Occupancy: {occupancy}")
print(f"Unique IDs Tracked: {len(vehicle_tracker.feature_history)}")
print("=" * 25)