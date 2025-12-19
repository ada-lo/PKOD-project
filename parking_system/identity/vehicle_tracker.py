import cv2 as cv
import numpy as np
from collections import defaultdict
import time
import config

class VehicleTracker:
    """Enhanced tracking with ReID features"""
    def __init__(self, tracklet_buffer=None):
        self.feature_history = defaultdict(list)  # {id: [feature_vectors]}
        self.position_history = defaultdict(list)  # {id: [(x,y)]}
        self.lost_tracks = {}  # {id: frames_lost}
        self.id_mapping = {}  # Map old IDs to recovered IDs
        self.tracklet_buffer = tracklet_buffer
        
    def extract_features(self, frame, box):
        """Extract simple visual features from bbox region"""
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
            
        # Resize to fixed size and compute color histogram
        roi_resized = cv.resize(roi, (64, 64))
        
        # HSV color histogram (more robust to lighting)
        hsv = cv.cvtColor(roi_resized, cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv.normalize(hist, hist).flatten()
        
        return hist
    
    def cosine_similarity(self, feat1, feat2):
        """Calculate cosine similarity between features"""
        if feat1 is None or feat2 is None:
            return 0.0
        dot_product = np.dot(feat1, feat2)
        norm_product = np.linalg.norm(feat1) * np.linalg.norm(feat2)
        return dot_product / (norm_product + 1e-6)
    
    def update(self, frame, track_id, box, center):
        """Update tracking history with ReID features"""
        # Extract features
        features = self.extract_features(frame, box)
        
        if features is not None:
            self.feature_history[track_id].append(features)
            # Keep only recent history
            if len(self.feature_history[track_id]) > config.REID_HISTORY_SIZE:
                self.feature_history[track_id].pop(0)
        
        # Store position
        self.position_history[track_id].append(center)
        if len(self.position_history[track_id]) > config.REID_HISTORY_SIZE:
            self.position_history[track_id].pop(0)
    
    def match_lost_track(self, frame, new_id, box, center):
        """Try to match a new ID with a recently lost track using ReID"""
        # If tracker already known, nothing to do
        if new_id in self.feature_history:
            return new_id

        # First, try time-based tracklet re-association if available
        if self.tracklet_buffer is not None:
            now = time.time()
            match = self.tracklet_buffer.reassociate(center, now)
            if match is not None:
                # map new_id to matched canonical id
                self.id_mapping[new_id] = match
                return match

        if not config.USE_REID:
            return new_id
        
        features = self.extract_features(frame, box)
        if features is None:
            return new_id
        
        best_match_id = None
        best_similarity = config.REID_SIMILARITY_THRESH
        
        # Compare with recently lost tracks
        for lost_id, frames_lost in list(self.lost_tracks.items()):
            if frames_lost > config.TRACK_BUFFER:
                del self.lost_tracks[lost_id]
                continue
            
            if lost_id not in self.feature_history:
                continue
            
            # Calculate average similarity with historical features
            similarities = []
            for hist_feat in self.feature_history[lost_id][-5:]:  # Last 5 frames
                sim = self.cosine_similarity(features, hist_feat)
                similarities.append(sim)
            
            avg_sim = np.mean(similarities) if similarities else 0.0
            
            # Also check spatial proximity
            if self.position_history[lost_id]:
                last_pos = self.position_history[lost_id][-1]
                distance = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                
                # Penalize if too far away (unlikely to be same vehicle)
                if distance > 200:
                    avg_sim *= 0.5
            
            if avg_sim > best_similarity:
                best_similarity = avg_sim
                best_match_id = lost_id
        
        if best_match_id is not None:
            # Recovered lost track!
            self.id_mapping[new_id] = best_match_id
            del self.lost_tracks[best_match_id]
            return best_match_id
        
        return new_id
    
    def mark_present_ids(self, current_ids):
        """Update lost track counter"""
        # Increment lost counter for all previous IDs
        for track_id in list(self.lost_tracks.keys()):
            self.lost_tracks[track_id] += 1
        
        # Mark currently visible IDs as present
        for track_id in current_ids:
            if track_id in self.lost_tracks:
                del self.lost_tracks[track_id]
        
        # Add newly lost tracks
        all_known_ids = set(self.feature_history.keys())
        current_ids_set = set(current_ids)
        newly_lost = all_known_ids - current_ids_set
        
        for track_id in newly_lost:
            if track_id not in self.lost_tracks:
                self.lost_tracks[track_id] = 1
                # Push tracklet into buffer for re-association
                if self.tracklet_buffer is not None:
                    feats = list(self.feature_history.get(track_id, []))
                    poss = list(self.position_history.get(track_id, []))
                    self.tracklet_buffer.add_lost(track_id, feats, poss, time.time())