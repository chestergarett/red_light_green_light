import cv2

class Tracker:
    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()
    
    def initialize_tracker(self, frame, bbox):
        self.tracker.init(frame, bbox)
    
    def update_tracker(self, frame):
        success, bbox = self.tracker.update(frame)
        return success, bbox

