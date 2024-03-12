import librosa
import numpy as np
import joblib

# Function to extract MFCC features using librosa
def extract_mfcc_librosa(audio_file_path):
    try:
        print("Extracting")
        # Load audio file
        audio, _ = librosa.load(audio_file_path, res_type='kaiser_fast')

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13)

        # Take the mean of each MFCC coefficient over time
        mfccs_mean = np.mean(mfccs, axis=1)
        print("Extracted")
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None

def predict_depression(audio_file_path, model_path, threshold=0.5):
    # Load the saved model
    print("here in prdict depression")
    rf_classifier = joblib.load(model_path)

    # Extract MFCC features from the audio file
    features = extract_mfcc_librosa(audio_file_path)

    # Check if extraction was successful
    if features is not None:
        # Reshape features to match the model input shape
        features = features.reshape(1, -1)

        try:
            # Use the trained model for decision scores
            decision_scores = rf_classifier.decision_function(features)
        except AttributeError:
            # If decision_function is not available, use predict_proba
            try:
                decision_scores = rf_classifier.predict_proba(features)[:, 1]
            except AttributeError:
                raise AttributeError("The model does not have decision_function or predict_proba.")

        # Adjust the threshold based on your model characteristics
        prediction = 1 if decision_scores > threshold else 0

        return prediction  # Return the predicted label (0 or 1)
    else:
        return None
