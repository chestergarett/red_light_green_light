from ultralytics import YOLO

class RedLightGreenLight:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
    
    def detect_players(self, frame, target_class_id=0, confidence_threshold=0.8):
        results = self.model(frame)
        detections = []

        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]
            confidence = result.conf[0]
            class_id = result.cls[0]

            if int(class_id) == target_class_id and confidence >= confidence_threshold:
                detections.append({
                    'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                    'confidence': confidence.item(),
                    'class_id': int(class_id)
                })
        
        return detections