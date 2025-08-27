from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import torch
import tensorflow as tf
from models.classifiers import Meso4, ResNet50Model, XceptionModel
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

# Load models once at startup
global_meso4 = Meso4()
global_meso4.load('models/weights/Meso4_DF.h5')

# Initialize ResNet50 and Xception models (they use pre-trained weights)
global_resnet50 = ResNet50Model()
global_xception = XceptionModel()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file with different models
        results = process_file(filepath)
        
        return jsonify(results)
    
    return jsonify({'error': 'Invalid file type'}), 400


def extract_frame_from_video(video_path):
    """Extract a frame from video for analysis"""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        return None
        
    # Convert BGR to RGB and save as temp image
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_frame.jpg')
    cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return temp_path

def process_file(filepath):
    # Check if file is video
    is_video = filepath.lower().endswith(('.mp4', '.avi'))
    temp_file = None
    
    try:
        # If it's a video, extract a frame for analysis
        if is_video:
            temp_file = extract_frame_from_video(filepath)
            if not temp_file:
                return {'error': 'Could not extract frame from video'}
            analysis_path = temp_file
        else:
            analysis_path = filepath
            
        # Process with all models
        results = {
            'DeepFaceLab': analyze_deepfacelab(analysis_path),
            'DFDNet': analyze_dfdnet(analysis_path),
            'MesoNet': analyze_mesonet(analysis_path),
            'FaceForensics': analyze_faceforensics(analysis_path),
            'ResNet50': analyze_resnet50(analysis_path),
            'Xception': analyze_xception(analysis_path)
        }
        
        # Add video flag to results
        if is_video:
            for key in results:
                if isinstance(results[key], dict):
                    results[key]['is_video'] = True
        
        return results
        
    finally:
        # Clean up temp file if created
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

def analyze_deepfacelab(filepath):
    # Placeholder for DeepFaceLab analysis
    # TODO: Implement actual model integration
    return {'score': 0.85, 'is_fake': True}

def analyze_dfdnet(filepath):
    # Placeholder for DFDNet analysis
    # TODO: Implement actual model integration
    return {'score': 0.92, 'is_fake': True}

def analyze_mesonet(filepath):
    try:
        # Handle both image and video frames
        if filepath.lower().endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(filepath)
            success, frame = cap.read()
            cap.release()
            
            if not success:
                return {'score': 0.0, 'is_fake': None, 'error': 'Could not read video frame'}
                
            # Convert BGR to RGB and resize
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            x = img.astype('float32') / 255.0
        else:
            # Handle image file
            img = keras_image.load_img(filepath, target_size=(256, 256))
            x = keras_image.img_to_array(img) / 255.0
            
        x = np.expand_dims(x, axis=0)
        pred = global_meso4.predict(x)[0][0]
        is_fake = pred > 0.5
        return {'score': float(pred), 'is_fake': bool(is_fake)}
        
    except Exception as e:
        return {'score': 0.0, 'is_fake': None, 'error': str(e)}

def analyze_faceforensics(filepath):
    # Placeholder for FaceForensics++ analysis
    # TODO: Implement actual model integration
    return {'score': 0.88, 'is_fake': True}

def analyze_resnet50(filepath):
    # Only process images for now
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        return {'score': 0.0, 'is_fake': None, 'error': 'Only image files are supported for ResNet50'}
    try:
        img = keras_image.load_img(filepath, target_size=(256, 256))
        x = keras_image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        pred = global_resnet50.predict(x)[0][0]
        is_fake = pred > 0.5
        return {'score': float(pred), 'is_fake': bool(is_fake)}
    except Exception as e:
        return {'score': 0.0, 'is_fake': None, 'error': str(e)}

def analyze_xception(filepath):
    # Only process images for now
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        return {'score': 0.0, 'is_fake': None, 'error': 'Only image files are supported for Xception'}
    try:
        img = keras_image.load_img(filepath, target_size=(256, 256))
        x = keras_image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        pred = global_xception.predict(x)[0][0]
        is_fake = pred > 0.5
        return {'score': float(pred), 'is_fake': bool(is_fake)}
    except Exception as e:
        return {'score': 0.0, 'is_fake': None, 'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True)