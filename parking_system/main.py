import cv2 as cv
import time
import os
import config
from capture.stream import open_capture, _is_stream
from tracking.detector import Detector
from identity.vehicle_tracker import VehicleTracker
from identity.tracklet_buffer import TrackletBuffer
from events.event_manager import VehicleEventManager
from state.occupancy_store import load_occupancy, save_occupancy
from ui.overlay import draw_bounding_box, draw_counting_line, draw_ui_overlay, draw_full_message

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

# Load persisted occupancy
occupancy, entry_count, exit_count, last_update = load_occupancy()

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
    frame = cv.resize(frame, (config.TARGET_WIDTH, config.TARGET_HEIGHT))

    # Draw counting line
    draw_counting_line(frame)

    # Run detection
    results = detector.detect(frame)
    
    if results:
        boxes, ids, confs = detector.extract_detections(results)
        
        if len(boxes) > 0:
            current_frame_ids = []
            
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

                # Draw bounding box
                is_reid = track_id != original_id
                draw_bounding_box(frame, box, track_id, conf, is_reid=is_reid)

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
                                        save_occupancy(occupancy, entry_count, exit_count, reason='entry')
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
                                    save_occupancy(occupancy, entry_count, exit_count, reason='exit')

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

    # Draw UI overlay
    warmup_remaining = max(0, int(warmup_end - time.time()))
    lost_tracks_count = len(vehicle_tracker.lost_tracks)
    draw_ui_overlay(frame, occupancy, entry_count, exit_count, frozen, 
                   warmup_remaining, lost_tracks_count)

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
        save_occupancy(occupancy, entry_count, exit_count, reason='manual_reset')
        print("[MANUAL] Occupancy reset to 0 by operator")
    elif k == ord('s'):
        # Manual set occupancy value via console prompt
        try:
            val = int(input("Set occupancy to (integer): "))
            val = max(0, min(val, config.MAX_CAPACITY))
            occupancy = val
            frozen = False
            save_occupancy(occupancy, entry_count, exit_count, reason='manual_set')
            print(f"[MANUAL] Occupancy set to {occupancy} by operator")
        except Exception as e:
            print(f"Invalid input for occupancy set: {e}")

# Cleanup
vid.release()
cv.destroyAllWindows()

# Save final state
save_occupancy(occupancy, entry_count, exit_count)

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