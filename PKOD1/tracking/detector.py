from ultralytics import YOLO
import torch
import time
from tracking.tracker_config import create_tracker_config

class Detector:
    def __init__(self, model_path='path/to/yolo11m.pt', min_confidence=0.4):
        print("Loading YOLOv11 model...")
        try:
            self.model = YOLO(model_path)
            print("✓ YOLOv11m loaded successfully")
        except Exception:
            print("YOLOv11 not found, falling back to YOLOv8x...")
            self.model = YOLO('yolov8x.pt')
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        print(f"Running on: {device.upper()}")
        
        self.min_confidence = min_confidence
        self.last_infer_time = 0.0
        self.target_infer_fps = 10.0
        self.tracker_config = create_tracker_config()
    
    def detect(self, frame, force_infer=False):
        """Run detection if enough time has passed since last inference."""
        now = time.time()
        if not force_infer and (now - self.last_infer_time < 1.0 / self.target_infer_fps):
            return None
        
        self.last_infer_time = now
        try:
            results = self.model.track(
                frame,
                persist=True,
                classes=[2],  # CARS ONLY
                conf=self.min_confidence,
                verbose=False,
                tracker=self.tracker_config
            )
            return results
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def extract_detections(self, results):
        """Extract boxes, IDs, and confidences from results."""
        if results is None or not hasattr(results[0].boxes, 'id') or results[0].boxes.id is None:
            return [], [], []
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        
        return boxes, ids, confs