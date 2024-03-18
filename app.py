import os
import time
import subprocess
import librosa
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify
from io import BytesIO
import random

app = Flask(__name__)

MODEL_PATH = r'C:\Users\kshit\Downloads\amjor\dep\impl\frontend\Project\model_processed.joblib'  # Path to your trained model

# Load the trained model
rf_classifier = joblib.load(MODEL_PATH)

# Function to extract MFCC features using librosa
def extract_mfcc(audio_data, sample_rate, n_fft=512):
    try:
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, n_fft=n_fft)

        # Take the mean of each MFCC coefficient over time
        mfccs_mean = np.mean(mfccs, axis=1)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing audio data: {e}")
        return None

def predict_audio(audio_data, sample_rate):
    # Extract features and predict using the model
    features = extract_mfcc(audio_data, sample_rate)
    if features is not None:
        features = features.reshape(1, -1)
        decision_scores = rf_classifier.predict_proba(features)[:, 1]
        prediction = 1 if decision_scores < 0.62 else 0
        if prediction == 1:
            decision_score = decision_scores[0]  # Get the single float value from the NumPy array
            # return f"Depression Found, Depression rate: {decision_score:.2f}%"
            return "Depression Found!"
        else:
            decision_score = decision_scores[0]
            # return f"Depression not found!  {decision_score:.2f}%"
            return "Depression not found!"
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'})

    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read audio file into memory
        audio_data, sample_rate = librosa.load(BytesIO(audio_file.read()), res_type='kaiser_fast')
        
        # Perform prediction
        prediction = predict_audio(audio_data, sample_rate)
        
        if prediction is not None:
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Prediction failed'})
    except Exception as e:
        return jsonify({'error': f'Error processing audio data: {e}'})

if __name__ == '__main__':
    app.run(debug=True)
