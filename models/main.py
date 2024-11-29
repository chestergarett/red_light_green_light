import cv2
from motion import MotionAnalyzer
from tracker import Tracker
from red_light_green_light import RedLightGreenLight
import mediapipe as mp
import json

pixel_to_meter_total_distance = 10 / 1200 # numerator in meters, denominator frame width in pixels of video  -> getting the width of
pixel_to_meter_player = 1.8 / 200 # numerator is height in meters, denominator is bounding box of player height in pixels

def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    model = RedLightGreenLight()
    cap = cv2.VideoCapture(video_path)
    analyzer = MotionAnalyzer(pixel_to_meter_total_distance,pixel_to_meter_player)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    detections_main = []
    poses = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)


        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        poses[timestamp] = {}
        detections = model.detect_players(frame)
        detections_main.append(detections)
        for i, detection in enumerate(sorted(detections, key=lambda x: x['bbox'][0])):
            bbox = detection['bbox']
            player_id = i
            analyzer.update(player_id, bbox, timestamp)

            if pose_results.pose_landmarks:
                keypoints = {}
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    keypoints[f"keypoint_{idx}"] = (landmark.x * frame.shape[1], landmark.y * frame.shape[0])

                analyzer.update_pose(player_id, keypoints, timestamp)
                pose_description = analyzer.analyze_pose(player_id)
                poses[timestamp][player_id] = pose_description

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    players = []
    for player_id in analyzer.positions:
        player_id = player_id
        total_distance_meters = round(analyzer.calculate_distance(player_id),2)
        ave_speed = round(analyzer.calculate_average_speed(player_id),2)
        deceleration_rate = round(analyzer.calculate_deceleration(player_id),2)
        stop_count = analyzer.get_stop_count(player_id)
        move_count = analyzer.get_move_count(player_id)
        motion_pattern = analyzer.calculate_motion_pattern(player_id)

        players.append({'player_id': player_id, 
                        'total_distance_meters': total_distance_meters, 
                        'ave_speed_m_s': ave_speed, 
                        'deceleration_rate_m_s': deceleration_rate,
                        'stop_count': stop_count,
                        'move_count': move_count,
                        'motion_rate_variation': motion_pattern['motion_rate_variation'],
                        'distance_per_motion': motion_pattern['distance_per_motion']
        })

    with open(f'outputs/player_metrics.json', 'w') as json_file:
        json.dump(players, json_file)

    with open(f'outputs/player_pose_analysis.json', 'w') as json_file:
        json.dump(poses, json_file)    

    with open(f'outputs/detections.json', 'w') as json_file:
        json.dump(detections_main, json_file)    
    
if __name__ == "__main__":
    process_video(f"training/training-video-001.mp4")

