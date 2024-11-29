import numpy as np

class MotionAnalyzer:
    def __init__(self, pixel_to_meter_distance, pixel_to_meter_player, stop_threshold=0.1):
        self.pixel_to_meter_distance = pixel_to_meter_distance
        self.positions = {} 
        self.speeds = {} 
        self.pixel_to_meter_player = pixel_to_meter_player
        self.stop_threshold = stop_threshold 
        self.states = {}  
        self.stop_counts = {} 
        self.move_counts = {}  
        self.last_processed_second = {} 
        self.keypoints = {}
        self.motion_distances = {}
        
    def update(self, player_id, bbox, timestamp):
        current_second = int(timestamp)

        if self.last_processed_second.get(player_id, -1) == current_second:
            return  # Skip processing if we're still in the same second

        self.last_processed_second[player_id] = current_second

        # Calculate the centroid of the bounding box
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

        if player_id not in self.positions:
            # Initialize data for new player
            self.positions[player_id] = [(timestamp, cx, cy)]
            self.speeds[player_id] = []
            self.states[player_id] = "stopped"
            self.stop_counts[player_id] = 0
            self.move_counts[player_id] = 0
            self.motion_distances[player_id] = 0
        else:
            # Update positions and calculate speed in pixels
            last_timestamp, last_cx, last_cy = self.positions[player_id][-1]
            distance_pixels = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
            time_diff = timestamp - last_timestamp

            # Convert speed to meters per second
            speed_pixels_per_second = distance_pixels / time_diff if time_diff > 0 else 0
            speed_meters_per_second = speed_pixels_per_second * self.pixel_to_meter_player

            self.speeds[player_id].append(speed_meters_per_second)
            self.positions[player_id].append((timestamp, cx, cy))

            # Detect state transitions
            current_state = self.states[player_id]
            if speed_meters_per_second < self.stop_threshold:
                # Player is considered "stopped"
                if current_state == "moving":
                    self.states[player_id] = "stopped"
                    self.stop_counts[player_id] += 1
            else:
                # Player is considered "moving"
                if current_state == "stopped":
                    self.states[player_id] = "moving"
                    self.move_counts[player_id] += 1

                # Accumulate motion distance
                self.motion_distances[player_id] += distance_pixels * self.pixel_to_meter_distance

    def update_pose(self, player_id, keypoints, timestamp):
        """
        Update player pose based on detected keypoints and timestamp.
        """
        if player_id not in self.keypoints:
            self.keypoints[player_id] = []
        
        # Store the keypoints for this frame
        self.keypoints[player_id].append((timestamp, keypoints))

    def analyze_pose(self, player_id):
        if player_id not in self.keypoints or not self.keypoints[player_id]:
            return "Unknown"
        
        _, keypoints = self.keypoints[player_id][-1]

        if 'left_knee' in keypoints and 'left_ankle' in keypoints:
            knee = keypoints['left_knee']
            ankle = keypoints['left_ankle']

            if abs(knee[1] - ankle[1]) < 50:
                return "Crouching"

        if 'head' in keypoints and 'feet' in keypoints:
            head = keypoints['head']
            feet = keypoints['feet']

            if abs(head[0] - feet[0]) < 50:
                return "Standing"
            elif abs(head[1] - feet[1]) < 100:
                return "Lying Down"

        return "Moving"

    def calculate_distance(self, player_id):
        if player_id not in self.positions:
            return 0
        positions = self.positions[player_id]
        distance_in_pixels = sum(
            np.sqrt((positions[i][1] - positions[i - 1][1])**2 + 
                    (positions[i][2] - positions[i - 1][2])**2)
            for i in range(1, len(positions))
        )
        
        # Convert distance to meters
        distance_in_meters = distance_in_pixels * self.pixel_to_meter_distance
        return distance_in_meters
    
    def calculate_average_speed(self, player_id):
        speeds = self.speeds.get(player_id, [])
        return np.mean(speeds) if speeds else 0
    
    def calculate_motion_pattern(self, player_id):
        """
        Analyze motion patterns such as variation in motion rate and distance covered per motion.
        """
        speeds = self.speeds.get(player_id, [])
        total_distance = self.motion_distances.get(player_id, 0)
        move_count = self.move_counts.get(player_id, 0)

        # Variation in motion rate (standard deviation of speeds)
        motion_rate_variation = np.std(speeds) if speeds else 0

        # Distance covered per motion event
        distance_per_motion = total_distance / move_count if move_count > 0 else 0

        return {
            "motion_rate_variation": motion_rate_variation,
            "distance_per_motion": distance_per_motion,
        }
    
    def calculate_deceleration(self, player_id):
        speeds = self.speeds.get(player_id, [])
        timestamps = [timestamp for timestamp, _, _ in self.positions.get(player_id, [])]  # Get corresponding timestamps
        
        decelerations = []
        for i in range(1, len(speeds)):  # Start from the second element
            speed_diff = speeds[i - 1] - speeds[i]  # Speed difference (in m/s)
            time_diff = timestamps[i] - timestamps[i - 1]  # Time difference (in seconds)
            
            if time_diff > 0:
                deceleration = speed_diff / time_diff  # Deceleration (in m/s²)
                decelerations.append(deceleration)
        
        return np.mean(decelerations) if decelerations else 0
    
    def get_stop_count(self, player_id):
        return self.stop_counts.get(player_id, 0)
    
    def get_move_count(self, player_id):
        return self.move_counts.get(player_id, 0)

    def get_current_state(self, player_id):
        return self.states.get(player_id, "stopped")
