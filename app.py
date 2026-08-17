import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from mtcnn import MTCNN
import shutil

# --- Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
TEMP_FACES_FOLDER = 'temp_faces_for_app'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# --- Load Your Best Model ---
try:
    classifier_model = tf.keras.models.load_model('classifier_model_robust.h5')
    detector = MTCNN()
    print("✅ Robust classifier model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier_model = None

def extract_faces_from_video(video_path, output_folder, faces_per_video=20):
    """
    Extracts face images from a single video file.
    Based on the logic from prepare_dataset.py
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // faces_per_video)
    frame_num, saved_count = 0, 0
    
    while saved_count < faces_per_video:
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
            
            face_filename = f"face_{saved_count:02d}.jpg"
            cv2.imwrite(os.path.join(output_folder, face_filename), face)
            saved_count += 1
        
        frame_num += frame_interval
    cap.release()
    return saved_count

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files or request.files['video'].filename == '':
            return "No video file selected"
        
        file = request.files['video']
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)

        # --- Prediction Logic ---
        if classifier_model is None:
            return "Model not loaded", 0.0

        saved_faces_count = extract_faces_from_video(video_path, TEMP_FACES_FOLDER)
        if saved_faces_count == 0:
            result, confidence = "No face detected", 0.0
        else:
            face_images = []
            for i in range(saved_faces_count):
                img_path = os.path.join(TEMP_FACES_FOLDER, f"face_{i:02d}.jpg")
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                face_images.append(img_array)
            
            face_images = np.array(face_images) / 255.0
            predictions = classifier_model.predict(face_images)
            avg_prediction = np.mean(predictions)
            
            # The generator labels folders alphabetically: fake=0, real=1
            if avg_prediction > 0.5:
                result = "REAL"
                confidence = avg_prediction * 100
            else:
                result = "FAKE"
                confidence = (1 - avg_prediction) * 100

        os.remove(video_path)
        return render_template('result.html', result=result, confidence=f"{confidence:.2f}")

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)