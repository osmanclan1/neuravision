import cv2
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
import pandas as pd

# Load YOLO model
def load_yolo_model(config_path, weights_path, names_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

# Detect objects
def detect_objects(img, net, output_layers):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids, confidences, boxes

# Draw bounding boxes
def draw_bounding_boxes(img, class_ids, confidences, boxes, classes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Exclude non-relevant objects
def exclude_non_relevant_objects(class_ids, confidences, boxes, relevant_classes):
    relevant_indices = [i for i, class_id in enumerate(class_ids) if class_id in relevant_classes]
    return [class_ids[i] for i in relevant_indices], [confidences[i] for i in relevant_indices], [boxes[i] for i in relevant_indices]

# Process each frame
def process_frame(frame, net, output_layers, classes, relevant_classes):
    class_ids, confidences, boxes = detect_objects(frame, net, output_layers)
    class_ids, confidences, boxes = exclude_non_relevant_objects(class_ids, confidences, boxes, relevant_classes)
    draw_bounding_boxes(frame, class_ids, confidences, boxes, classes)
    return frame, class_ids, confidences, boxes

# Initialize trackers
def initialize_trackers(boxes, frame):
    trackers = []
    for box in boxes:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, tuple(box))
        trackers.append(tracker)
    return trackers

# Update trackers
def update_trackers(trackers, frame):
    tracked_boxes = []
    for tracker in trackers:
        success, box = tracker.update(frame)
        if success:
            tracked_boxes.append(box)
    return tracked_boxes

# Track ball
def track_ball(class_ids, confidences, boxes):
    for i, class_id in enumerate(class_ids):
        if class_id == 32:  # Assuming 32 is the class ID for the ball
            return {"bbox": boxes[i], "confidence": confidences[i]}
    return None

# Track players
def track_players(class_ids, confidences, boxes):
    players = []
    for i, class_id in enumerate(class_ids):
        if class_id == 0:  # Assuming 0 is the class ID for persons
            players.append({"bbox": boxes[i], "confidence": confidences[i]})
    return players

# Assign ball to closest player
def assign_ball_to_player(ball, players):
    if ball is None:
        return
    ball_center = calculate_center(ball['bbox'])
    closest_player = min(players, key=lambda p: np.linalg.norm(np.array(calculate_center(p['bbox'])) - np.array(ball_center)))
    closest_player['has_ball'] = True

# Calculate center of bounding box
def calculate_center(bbox):
    x, y, w, h = bbox
    return x + w // 2, y + h // 2

# Define triangle points based on bounding box
def get_triangle_points(bbox):
    center = calculate_center(bbox)
    points = [center, (center[0] + 10, center[1] + 10), (center[0] - 10, center[1] + 10)]
    return points

# Segment image using KMeans
def segment_image(frame):
    reshaped_frame = frame.reshape((-1, 3))
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(reshaped_frame)
    segmented_img = labels.reshape(frame.shape[:2])
    return segmented_img

# Analyze clusters
def analyze_clusters(segmented_img):
    # Implement color analysis to differentiate players and non-players
    pass

# Detect player colors
def detect_player_colors(frame):
    segmented_img = segment_image(frame)
    return analyze_clusters(segmented_img)

# Identify teams
def identify_teams(tracked_objects):
    for obj in tracked_objects:
        team_color = detect_player_colors(obj['image'])
        obj['team'] = team_color

# Interpolate missing values
def interpolate_missing_values(data):
    df = pd.DataFrame(data)
    df.interpolate(method='linear', inplace=True)
    return df.to_dict()

# Track ball and assign to players
def track_ball_and_assign(frame, net, output_layers, ball_tracker, player_trackers):
    class_ids, confidences, boxes = detect_objects(frame, net, output_layers)
    ball = track_ball(class_ids, confidences, boxes)
    players = track_players(class_ids, confidences, boxes)

    if ball and not ball_tracker:
        ball_tracker = initialize_trackers([ball['bbox']], frame)[0]

    if not player_trackers and players:
        player_trackers = initialize_trackers([p['bbox'] for p in players], frame)

    if ball_tracker:
        ball['bbox'] = update_trackers([ball_tracker], frame)[0]

    if player_trackers:
        player_boxes = update_trackers(player_trackers, frame)
        for i, player in enumerate(players):
            player['bbox'] = player_boxes[i]

    assign_ball_to_player(ball, players)
    return ball_tracker, player_trackers

# Draw transparent rectangle
def draw_transparent_rect(frame, bbox):
    overlay = frame.copy()
    cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# Calculate ball possession
def calculate_ball_possession(tracked_data):
    possession = {'team1': 0, 'team2': 0}
    for frame_data in tracked_data:
        if frame_data['player']['has_ball']:
            possession[frame_data['player']['team']] += 1
    total_frames = sum(possession.values())
    possession_percentage = {team: (time / total_frames) * 100 for team, time in possession.items()}
    return possession_percentage

# Detect camera motion
def detect_camera_motion(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, corners, None)
    return next_corners - corners

# Adjust positions for camera movement
def adjust_positions_for_camera_movement(tracked_objects, camera_movement):
    for obj in tracked_objects:
        obj['adjusted_position'] = obj['position'] - camera_movement

# Convert position to real-world coordinates
def convert_to_real_world(position, scale_factor):
    return position * scale_factor

# Perspective transform
def perspective_transform(point, matrix):
    transformed_point = cv2.perspectiveTransform(np.array([[point]], dtype='float32'), matrix)
    return transformed_point[0][0]

# Estimate speed and distance
def estimate_speed(distance, time_interval):
    return distance / time_interval

def estimate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Main function
def main(input_video, output_video, config_path, weights_path, names_path, relevant_classes):
    net, output_layers, classes = load_yolo_model(config_path, weights_path, names_path)
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 30, (1280, 720))
    
    ball_tracker = None
    player_trackers = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, class_ids, confidences, boxes = process_frame(frame, net, output_layers, classes, relevant_classes)
        ball_tracker, player_trackers = track_ball_and_assign(processed_frame, net, output_layers, ball_tracker, player_trackers)
        
        out.write(processed_frame)
        cv2.imshow('Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = '/Users/yacoubosman/Downloads/Movie.mp4'  # Update the input video path
    output_video = 'output.avi'
    config_path = 'yolov3.cfg'
    weights_path = 'yolov3.weights'
    names_path = 'coco.names.txt'
    relevant_classes = [0]  # Class ID for person in COCO dataset
    main(input_video, output_video, config_path, weights_path, names_path, relevant_classes)
