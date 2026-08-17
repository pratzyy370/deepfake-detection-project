import os
import cv2
from mtcnn import MTCNN
import shutil

# --- Configuration ---
SOURCE_DIR = 'final_dataset'
OUTPUT_DIR = 'processed_master_dataset'
FACES_PER_VIDEO = 20

def process_all_videos():
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing old '{OUTPUT_DIR}' directory...")
        shutil.rmtree(OUTPUT_DIR)
    detector = MTCNN()
    print("--- Starting master dataset processing ---")
    for set_name in ['train', 'validation', 'test']:
        print(f"\nProcessing '{set_name}' set...")
        source_set_path = os.path.join(SOURCE_DIR, set_name)
        output_set_path = os.path.join(OUTPUT_DIR, set_name)
        for category in ['real', 'fake']:
            print(f"  Processing '{category}' videos...")
            source_category_path = os.path.join(source_set_path, category)
            output_category_path = os.path.join(output_set_path, category)
            os.makedirs(output_category_path, exist_ok=True)
            videos = [v for v in os.listdir(source_category_path) if v.endswith(('.mp4', '.mov', '.avi'))]
            for i, video_filename in enumerate(videos):
                video_path = os.path.join(source_category_path, video_filename)
                video_name_prefix = os.path.splitext(video_filename)[0]
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"    Warning: Could not open {video_filename}. Skipping.")
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_interval = max(1, total_frames // FACES_PER_VIDEO)
                frame_num = 0
                saved_count = 0
                while saved_count < FACES_PER_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret: break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detector.detect_faces(frame_rgb)
                    if results:
                        x, y, w, h = results[0]['box']
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                        face = frame[y1:y2, x1:x2]
                        face_filename = f"{video_name_prefix}_face_{saved_count:02d}.jpg"
                        cv2.imwrite(os.path.join(output_category_path, face_filename), face)
                        saved_count += 1
                    frame_num += frame_interval
                cap.release()
                if (i + 1) % 10 == 0:
                    print(f"    Processed {i+1}/{len(videos)} videos in '{category}' folder.")
    print("\n✅ Master dataset processing complete!")

if __name__ == "__main__":
    process_all_videos()