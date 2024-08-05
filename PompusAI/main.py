import numpy as np
import librosa
import speech_recognition as sr
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load and prepare the training data for speaker identification
def prepare_speaker_model():
    # Replace these with paths to your recorded training audio files
    training_files = ['person1.wav', 'person2.wav']
    # Labels corresponding to the speakers
    labels = [0, 1]  # Example labels for different speakers
    
    X_train = np.array([extract_features(file) for file in training_files])
    y_train = np.array(labels)
    
    model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    model.fit(X_train, y_train)
    return model

# Predict the speaker based on the audio file
def predict_speaker(file_path, model):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Recognize speech and identify the speaker
def process_audio(file_path, model):
    # Recognize speech
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio)
            print("Recognized Text:", text)
        except sr.UnknownValueError:
            print("Could not understand audio")
            return
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return
    
    # Identify speaker
    speaker_id = predict_speaker(file_path, model)
    print("Identified Speaker ID:", speaker_id)

# Prepare the speaker identification model
speaker_model = prepare_speaker_model()

# Process an audio file
audio_file_path = 'test_audio.wav'  # Replace with your test audio file path
process_audio(audio_file_path, speaker_model)
