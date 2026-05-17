import os
import json
from config import ALL_WORDS

dataset_path = "dataset/"
wlasl_json = os.path.join(dataset_path, "WLASL_v0.3.json")
videos_dir = os.path.join(dataset_path, "videos")

with open(wlasl_json, 'r') as f:
    wlasl_data = json.load(f)

# Build a set of video IDs that are in our target 1000 words
useful_videos = set()
for entry in wlasl_data:
    gloss = entry['gloss']
    if gloss in ALL_WORDS:
        for inst in entry['instances']:
            useful_videos.add(inst['video_id'] + ".mp4")

# Scan the actual videos directory
all_videos = set(f for f in os.listdir(videos_dir) if f.endswith('.mp4'))
unused_videos = all_videos - useful_videos

total_size_bytes = 0
for vid in unused_videos:
    vid_path = os.path.join(videos_dir, vid)
    total_size_bytes += os.path.getsize(vid_path)

print(f"Total videos in dataset: {len(all_videos)}")
print(f"Useful videos for 1000 words: {len(all_videos & useful_videos)}")
print(f"Unused (nonsense) videos: {len(unused_videos)}")
print(f"Space to be freed: {total_size_bytes / (1024*1024):.2f} MB")

with open("unused_videos.txt", "w") as f:
    for vid in unused_videos:
        f.write(vid + "\n")
