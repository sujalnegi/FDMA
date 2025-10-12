import cv2
import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file
from werkzeug.utils import secure_filename
import time
import zipfile
from io import BytesIO
from datetime import datetime
import base64
from flask import Flask, request, render_template, redirect, url_for, jsonify
import imagehash
from PIL import Image

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
OUTPUT_DIR = 'out-img'

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PROTOTXT_PATH = 'deploy.prototxt.txt'
CAFFEMODEL_PATH = 'res10_300x300_ssd_iter_140000.caffemodel'
CONFIDENCE_THRESHOLD = 0.5
MIN_CROP_DIMENSION = 80
BLUR_THRESHOLD = 150.

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
except Exception as e:
    print("Error loading model files:", e)
    net, face_cascade = None, None

def is_blurry(image, threshold=BLUR_THRESHOLD):
    if image is None or image.size == 0:
        return True, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var 

def recheck_face_validity(cropped_face_img, dnn_net, confidence_threshold=CONFIDENCE_THRESHOLD):
    if cropped_face_img is None or cropped_face_img.size == 0 or dnn_net is None:
        return False, 0.0
    
    blob = cv2.dnn.blobFromImage(cv2.resize(cropped_face_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob) 
    detections = dnn_net.forward()
    max_confidence = 0.0

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if(confidence > max_confidence):
            max_confidence = confidence
    return max_confidence >= confidence_threshold, max_confidence
    
def process_image_and_extract(image_path, filters):
    
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not read image."}

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50)
    )

    saved_faces_info = []

    session_id = str(int(time.time()))
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(session_output_dir, exist_ok=True) 

    for i, (x, y, w, h) in enumerate(faces):
        cropped_face = image[y:y+h, x:x+w]

        if 'resolution-check' in filters and (w < MIN_CROP_DIMENSION or h < MIN_CROP_DIMENSION):
            continue

        if 'blur-check' in filters:
            is_blurry_flag, _ = is_blurry(cropped_face, BLUR_THRESHOLD)
            if is_blurry_flag:
                continue

        if 'dnn-recheck' in filters:
            is_valid_face, _ = recheck_face_validity(cropped_face, net, CONFIDENCE_THRESHOLD)
            if not is_valid_face:
                continue
        
        face_filename = f'{session_id}_face_{i:04d}_cropped.jpg'
        
        full_output_path = os.path.join(session_output_dir, face_filename)
        cv2.imwrite(full_output_path, cropped_face)

        saved_faces_info.append(os.path.join(session_id, face_filename))
    return {"faces": saved_faces_info, "session_id": session_id}

def process_video_file_and_extract(video_path, filters, remove_duplicates=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}
    saved_faces_info = []
    session_id = str(int(time.time()))
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(session_output_dir, exist_ok=True)

    frame_count = 0
    face_count = 0
    seen_hashes = set()

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame_count += 1

        if frame_count %5 != 0:
            continue
        image = frame
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.05, minNeighbors=5, minSize=(50,50)
        )
        for(x, y, w, h) in faces:
            cropped_face = image[y:y+h, x:x+w]

            if remove_duplicates:
                pil_image = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                current_hash = imagehash.phash(pil_image)

                is_duplicate = False
                for seen_hash in seen_hashes:
                    if current_hash - seen_hash <5:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                seen_hashes.add(current_hash)
            if 'resolution-check' in filters and (w < MIN_CROP_DIMENSION or h < MIN_CROP_DIMENSION): 
                continue
            if 'blur-check' in filters:
                is_blurry_flag, _ = is_blurry(cropped_face, BLUR_THRESHOLD)
                if is_blurry_flag:    
                    continue

            if 'dnn-recheck' in filters:
                is_valid_face, _ = recheck_face_validity(cropped_face, net, CONFIDENCE_THRESHOLD)
                if not is_valid_face:
                    continue
            face_filename = f'{session_id}_face_{face_count:04d}_cropped.jpg'
            full_output_path = os.path.join(session_output_dir, face_filename)
            cv2.imwrite(full_output_path, cropped_face)

            saved_faces_info.append(os.path.join(session_id, face_filename))
            face_count += 1
    cap.release()
    return {"faces": saved_faces_info, "session_id": session_id}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    if 'image-upload' not in request.files:
        return redirect(request.url)
    
    file = request.files['image-upload']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    username = request.form.get('username', 'anonymous')

    filters_active = request.form.getlist('filter-options')

    filename = secure_filename(file.filename)
    unique_filename = f"{username}_{int(time.time())}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename) 

    file.save(file_path)

    results = process_image_and_extract(file_path, filters_active) 

    if "error" in results:
        return f"Error: {results['error']}"

    return redirect(url_for('results', session_id=results['session_id'], count=len(results['faces']))) 

@app.route('/results/<session_id>/<int:count>')
def results(session_id, count):
    
    face_files = [os.path.join(session_id, f) 
                  for f in os.listdir(os.path.join(OUTPUT_DIR, session_id)) 
                  if f.endswith('.jpg')] 

    face_files_url_safe = [f.replace(os.path.sep, '/') for f in face_files]

    return render_template('results.html',
                           face_files=face_files_url_safe, 
                           count=count,
                           session_id=session_id)

@app.route('/video_upload', methods=['POST'])
def video_upload():

    file = request.files['video-upload']
    if 'video-upload' not in request.files:
        return redirect(url_for('home')) # Redirect if no file field

    username = request.form.get('username', 'video_user')
    filters_active = request.form.getlist('filter-options')
    remove_duplicates = 'remove-duplicates' in request.form


    filename = secure_filename(file.filename)
    unique_filename = f"{username}_{int(time.time())}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename) 
    file.save(file_path)

    results = process_video_file_and_extract(file_path, filters_active, remove_duplicates) 

    os.remove(file_path)

    if "error" in results:
        return f"Error processing video: {results['error']}"

    return redirect(url_for('results', session_id=results['session_id'], count=len(results['faces'])))

@app.route('/video')
def video_page():
    return render_template('video_upload.html')

@app.route('/out-img/<session_id>/<filename>')
def serve_face(session_id, filename):
    full_path = os.path.join(OUTPUT_DIR, session_id)
    return send_from_directory(full_path, filename)

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/webcam_capture', methods=['POST'])
def webcam_capture():
    if 'video-blob' not in request.files:
        return redirect(url_for('home'))
    file = request.files['video-blob']

    filters_active = request.form.getlist('filter-options')

    if file.filename == '':
        return redirect(url_for('home'))

    filters_active = request.form.getlist('filter-options')
    remove_duplicates = 'remove-duplicates' in request.form

    filename = f"webcam_{int(time.time())}.webm"
    temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    file.save(temp_video_path)

    results = process_video_file_and_extract(temp_video_path, filters_active, remove_duplicates)

    os.remove(temp_video_path)

    if "error" in results:
        return f"Error processing webcam video: {results['error']}"

    return redirect(url_for('results', session_id=results['session_id'], count=len(results['faces'])))    

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/download_zip/<session_id>')
def download_zip(session_id):
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)

    if not os.path.exists(session_output_dir) or not os.listdir(session_output_dir):
        return "Error: No files found for this session.", 404
    
    memory_file = BytesIO()

    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(session_output_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')): 
                file_path = os.path.join(session_output_dir, filename)
                zf.write(file_path, arcname=filename)
    
    memory_file.seek(0)
    zip_filename = f'{session_id}_extracted_faces.zip'
    
    return send_file(
        memory_file,
        download_name=zip_filename,
        as_attachment=True,
        mimetype='application/zip'
    )
    

if __name__ == '__main__':
    app.secret_key = 'password'
    app.run(debug=True)
