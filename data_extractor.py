import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv
import json
import argparse
from tqdm import tqdm
from config import ALL_WORDS
from utils import extract_two_hands

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

data = []
dataset_path = "dataset/"

wlasl_json = os.path.join(dataset_path, "WLASL_v0.3.json")
videos_dir = os.path.join(dataset_path, "videos")

if os.path.exists(wlasl_json) and os.path.exists(videos_dir):
    with open(wlasl_json, 'r') as f:
        wlasl_data = json.load(f)
        
    video_to_gloss = {}
    for entry in wlasl_data:
        gloss = entry['gloss']
        for inst in entry['instances']:
            video_to_gloss[inst['video_id']] = gloss
            
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    if args.limit:
        video_files = video_files[:args.limit]
    
    for i, video_file in enumerate(tqdm(video_files, desc="Extracting Videos")):
        video_id = video_file.replace('.mp4', '')
        if video_id not in video_to_gloss:
            continue
            
        gloss = video_to_gloss[video_id]
        if gloss not in ALL_WORDS:
            continue
            
        video_path = os.path.join(videos_dir, video_file)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            target_frames = [total_frames // 4, total_frames // 2, (total_frames * 3) // 4]
            for frame_idx in target_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    
                    try:
                        detection_result = detector.detect(mp_image)
                        if getattr(detection_result, 'hand_landmarks', None):
                            landmarks = extract_two_hands(detection_result)
                            landmarks.append(gloss)
                            data.append(landmarks)
                    except Exception:
                        pass
        cap.release()

elif os.path.exists(dataset_path):
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path) or label == "videos":
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            
            try:
                mp_image = mp.Image.create_from_file(img_path)
                detection_result = detector.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    hand_landmarks = detection_result.hand_landmarks[0]
                    landmarks = []
                    for landmark in hand_landmarks:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    landmarks.append(label)
                    data.append(landmarks)
            except Exception:
                continue

if len(data) > 0:
    with open("asl_hand_landmarks.csv", "w", newline='') as f:
        writer = csv.writer(f)
        header = [f"L_{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']] + \
                 [f"R_{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
        writer.writerow(header)
        writer.writerows(data)
