import cv2 as cv
import cvzone
import config

def draw_bounding_box(frame, box, track_id, conf, color=(255, 0, 255), is_reid=False):
    """Draw bounding box with ID and confidence."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    
    # Use different color for ReID matches
    draw_color = (0, 255, 255) if is_reid else color
    
    # Draw corner rectangle
    cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=draw_color)
    
    # Draw label
    label = f'ID:{track_id} ({conf:.2f})'
    if is_reid:
        label += " [ReID]"
    cvzone.putTextRect(frame, label, (max(0, x1), max(35, y1)), 
                     scale=0.8, thickness=1, colorR=draw_color)
    
    # Draw center point
    cx, cy = x1 + w // 2, y1 + h // 2
    cv.circle(frame, (cx, cy), 5, draw_color, cv.FILLED)
    
    return cx, cy

def draw_counting_line(frame, line_color=(0, 0, 255), highlight_color=(0, 255, 0)):
    """Draw the counting line."""
    cv.line(frame, (config.LINE[0], config.LINE[1]), 
           (config.LINE[2], config.LINE[3]), line_color, 5)
    return highlight_color

def draw_ui_overlay(frame, occupancy, entry_count, exit_count, frozen=False, 
                   warmup_remaining=0, lost_tracks_count=0):
    """Draw all UI elements on frame."""
    # Status background
    cv.rectangle(frame, (0, 0), (400, 180), (0, 0, 0), -1)
    
    # Status color based on occupancy
    status_color = (0, 255, 0) if occupancy < config.MAX_CAPACITY else (0, 0, 255)
    
    # Main stats
    cv.putText(frame, f"Occupancy: {occupancy}/{config.MAX_CAPACITY}", (20, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv.putText(frame, f"Total In: {entry_count}", (20, 80), 
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.putText(frame, f"Total Out: {exit_count}", (20, 120), 
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.putText(frame, f"Tracker: {config.TRACKER_TYPE.upper()}", (20, 160), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    # ReID status
    if config.USE_REID:
        reid_text = f"ReID: {lost_tracks_count} lost"
        cv.putText(frame, reid_text, (config.TARGET_WIDTH - 250, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # Warm-up / frozen status
    if warmup_remaining > 0:
        cv.putText(frame, f"WARM-UP: {warmup_remaining}s", 
                   (config.TARGET_WIDTH - 250, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    if frozen:
        cv.putText(frame, "FROZEN - MANUAL RESET REQUIRED (press 'r')", 
                   (420, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def draw_full_message(frame, message, position=(20, 200), color=(0, 0, 255)):
    """Draw a full parking message."""
    cvzone.putTextRect(frame, message, position, scale=1.5, thickness=2, colorR=color)