import cv2
from ultralytics import YOLO
import numpy as np
import os
import time

import cv2
from ultralytics import YOLO
import numpy as np
import time
import math

# Input and output video paths
input_video = r"D:\safety\input\wh.mp4"
output_video = 'fall_output_with_timer_and_warning121.mp4'

# Load the YOLOv8x pose model for higher accuracy
model = YOLO('yolov8x-pose.pt')

fall_timers = {}  # Track fall start times
warning_flags = {}  # Track if warning has been triggered per person

def get_posture_color(keypoints, box):
    try:
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
    except IndexError:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0

        if aspect_ratio > 1.2:
            return (0, 0, 255), "Fallen (Box)"
        else:
            return (255, 0, 0), "Unknown"

    torso_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
    hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

    dx = hip_center[0] - torso_center[0]
    dy = hip_center[1] - torso_center[1]

    angle = abs(math.degrees(math.atan2(dy, dx)))

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height if height > 0 else 0

    if aspect_ratio > 1.2 or (angle < 45 or angle > 135):
        return (0, 0, 255), "Fallen"
    elif 45 <= angle <= 135:
        return (0, 255, 0), "Standing"

    return (255, 0, 0), "Unknown"

def draw_warning_sign(frame, center_x, center_y, size=100):
    points = np.array([
        [center_x, center_y - size // 2],
        [center_x - size // 2, center_y + size // 2],
        [center_x + size // 2, center_y + size // 2]
    ])
    cv2.drawContours(frame, [points], 0, (0, 0, 255), -1)  # Filled red triangle
    cv2.circle(frame, (center_x, center_y + size // 4), size // 10, (255, 255, 255), -1)  # White dot

# Video processing
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise Exception(f"Cannot open video file {input_video}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0
print("Starting video processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Accuracy improved: higher confidence, larger image size, lower IoU
    results = model.predict(frame, save=False, verbose=False, conf=0.5, iou=0.4, imgsz=960)
    annotated_frame = frame.copy()

    if results and results[0].keypoints is not None:
        keypoints_list = results[0].keypoints.data.cpu().numpy()
        boxes = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        for i, person_kpts in enumerate(keypoints_list):
            if i < len(boxes):
                box = boxes[i]
                x1, y1, x2, y2 = box

                color, posture = get_posture_color(person_kpts, box)

                if "Fallen" in posture:
                    if i not in fall_timers:
                        fall_timers[i] = time.time()
                        warning_flags[i] = False
                    else:
                        elapsed_time = time.time() - fall_timers[i]

                        # Show timer above the bounding box
                        cv2.putText(annotated_frame, f"{int(elapsed_time)} sec", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        if elapsed_time > 6:  # 6 seconds threshold
                            warning_flags[i] = True

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                else:
                    fall_timers.pop(i, None)
                    warning_flags.pop(i, None)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                for keypoint in person_kpts:
                    if keypoint[2] > 0.3:
                        cv2.circle(annotated_frame, (int(keypoint[0]), int(keypoint[1])), 4, color, -1)

                cv2.putText(annotated_frame, posture, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        num_persons = len(keypoints_list)
        cv2.putText(annotated_frame, f'Persons: {num_persons}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show warning sign if any person is fallen more than 6 seconds
    if any(warning_flags.values()):
        draw_warning_sign(annotated_frame, width // 2, height // 2)

    out.write(annotated_frame)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"Processing complete. Output saved to {output_video}")

output_video = r"D:\safety\fall22.mp4"

model = YOLO('yolov8l-pose.pt')

fall_timers = {}  # Maps person ID (box center) to [start_time, last_seen_position]
FALL_THRESHOLD_SECONDS = 10
IOU_DISTANCE_THRESHOLD = 50  # max pixel distance between same person across frames

def get_posture_color(keypoints):
    try:
        nose = keypoints[0][:2]
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
    except IndexError:
        return (0, 255, 0),"head_below_hip"

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2
    nose_y = nose[1]

    if abs(shoulder_y - hip_y) < 30 and nose_y > hip_y:
        return (0, 0, 255), "Falled_on_ground"
    elif hip_y - shoulder_y > 50:
        return (0, 255, 0), "person_Standing/ walking"
    else:
        return (255, 0, 0), "sitting"

def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def find_matching_id(center, fall_timers):
    for pid, (_, prev_center) in fall_timers.items():
        if np.linalg.norm(np.array(center) - np.array(prev_center)) < IOU_DISTANCE_THRESHOLD:
            return pid
    return None

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise Exception(f"Cannot open video file {input_video}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print("Starting video processing...")

frame_count = 0
next_person_id = 0  # used to assign new IDs

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, save=False, verbose=False)
    annotated_frame = frame.copy()

    if results and results[0].keypoints is not None:
        keypoints_list = results[0].keypoints.data.cpu().numpy()
        boxes = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        current_fall_ids = {}

        for i, person_kpts in enumerate(keypoints_list):
            if i >= len(boxes): continue

            confidence = person_kpts[:, 2].mean()
            if confidence < 0.3:
                continue  # skip low-confidence person

            box = boxes[i]
            x1, y1, x2, y2 = box
            center = get_center(box)
            color, posture = get_posture_color(person_kpts)

            match_id = find_matching_id(center, fall_timers)

            if posture == "Fallen":
                if match_id is not None:
                    fall_timers[match_id][1] = center
                    elapsed_time = time.time() - fall_timers[match_id][0]
                    if elapsed_time > FALL_THRESHOLD_SECONDS:
                        cv2.putText(annotated_frame, f"⚠ WARNING: Person {match_id} fallen {int(elapsed_time)}s",
                                    (int(width / 4), 60 + 30 * match_id), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                else:
                    fall_timers[next_person_id] = [time.time(), center]
                    match_id = next_person_id
                    next_person_id += 1
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            else:
                if match_id is not None:
                    del fall_timers[match_id]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            for keypoint in person_kpts:
                if keypoint[2] > 0.3:
                    cv2.circle(annotated_frame, (int(keypoint[0]), int(keypoint[1])), 4, color, -1)

            cv2.putText(annotated_frame, posture, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Remove people no longer seen
        current_ids = [find_matching_id(get_center(b), fall_timers) for b in boxes]
        for pid in list(fall_timers.keys()):
            if pid not in current_ids:
                del fall_timers[pid]

        cv2.putText(annotated_frame, f'Persons Detected: {len(keypoints_list)}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    out.write(annotated_frame)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()

print(f"✅ Processing complete. Output saved to {output_video}")
