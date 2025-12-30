from flask import Flask, request, render_template, redirect, url_for, jsonify, session
from flask_cors import CORS
import pandas as pd
import sqlite3
import librosa
import numpy as np
import os
import tensorflow as tf
import base64
import random


from sklearn.preprocessing import OneHotEncoder
from tensorflow.python import keras
from tensorflow.python.keras.models import Model, load_model

from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import librosa

# Initialize the Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:5000", "http://localhost:3000"])
app.secret_key = os.environ.get("secret_key")

app.config.update(
    SESSION_COOKIE_SAMESITE="None",  # Allows cross-site cookies
    SESSION_COOKIE_SECURE=True       
)

# Database configuration
DATABASE = 'users.db'

def init_db():
    dbms = sqlite3.connect(DATABASE)
    cursor = dbms.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    dbms.commit()
    dbms.close()

# Initialize the database
init_db()

# Load the pre-trained model
model = tf.keras.models.load_model('cnn.h5')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Spotify Init
client_id = os.environ.get("client_id")
client_secret = os.environ.get("client_secret")
redirect_uri = 'http://localhost:5000/callback'
scope = 'streaming,'

cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=scope,
    cache_handler=cache_handler,
    show_dialog=True
)

sp = Spotify(auth_manager=sp_oauth)

# Emotion labels
EMOTION_LABELS = ['Happy','Disgust','Fear','Angry','Neutral','Sad','Suprise']


EMOTION_TO_SONG_LABEL = {
    "Happy": 0,                     
    "Sad": 1, "Disgust": 1,        
    "Angry": 3, "Surprise": 3,      
    "Neutral": 2, "Fear": 2         
}


# Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            
            if not username or not email or not password:
                return jsonify({"success": False, "message": "All fields are required!"}), 400
            
            dbms = sqlite3.connect(DATABASE)
            cursor = dbms.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, password))
            dbms.commit()
            dbms.close()
            
            return jsonify({"success": True, "message": "User registered successfully!"}), 200
        except sqlite3.IntegrityError:
            return jsonify({"success": False, "message": "User already exists!"}), 400
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500
    
    return render_template('registration.html')


# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        dbms = sqlite3.connect(DATABASE)
        cursor = dbms.cursor()
        cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        dbms.close()
        if result and result[0] == password:
            session['username'] = username
            return redirect(url_for('home'))
    return render_template('minorproj.html')


# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# Home
@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    
    return redirect(url_for('audio_input'))


# Audio input
@app.route('/audio_input', methods=['GET', 'POST'])
def audio_input():

    if request.method == 'POST':
        try:
            # Extract Base64-encoded audio data
            audio_base64 = request.form['audio']

            # Decode Base64 string to binary audio
            audio_binary = base64.b64decode(audio_base64.split(",")[1])

            # Save to temporary file
            temp_audio_path = "temp_audio.wav"
            with open(temp_audio_path, "wb") as audio_file:
                audio_file.write(audio_binary)

            # Extract Features
            def extract_features(y, sr):
                # Contrast
                result = np.array([])
                stft = np.abs(librosa.stft(y))
                contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
                result = np.hstack((result, contrast))

                # Chroma_stft(pitch)
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
                result = np.hstack((result, chroma))

                # MFCC(echo)
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))

                # tonnetz
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T,axis=0)
                result = np.hstack((result, tonnetz))   

                # MelSpectogram
                mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
                result = np.hstack((result, mel))

                #zero crossing rate
                zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
                result=np.hstack((result, zcr))

                #root mean square
                rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
                result = np.hstack((result, rms))

                #spectraal_bandwidth
                BW = np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr).T,axis=0)
                result = np.hstack((result, BW))
    
                return result

            def get_features(path):
                # Load audio
                data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

                # Extract features
                res1 = extract_features(data, sample_rate)
                result = np.array(res1)

                return result
            
            features = get_features(temp_audio_path)

            features = features.reshape(1, -1)

            # Predict emotion using the model
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction[0])

            print(predicted_class)

            # Map the predicted class to the emotion label
            predicted_emotion = EMOTION_LABELS[predicted_class]

            # Remove temporary file
            #os.remove(temp_audio_path)

            print(predicted_emotion)

            session["detected_emotion"] = predicted_emotion

            return  {"emotion": predicted_emotion}
        except Exception as e:
            print(f"Error processing audio: {e}")
            return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

    return render_template('chatbot.html', detected_emotion="None")


#Callback
@app.route('/callback')
def callback():
    sp_oauth.get_access_token(request.args['code'])

    return redirect(url_for('home'))


#Get song URI
@app.route('/get_playlist')
def get_playlist():
    try:
        # URI generation
        df = pd.read_csv("URI.csv")
        detected_emotion = session.get("detected_emotion")
        song_label = EMOTION_TO_SONG_LABEL.get(detected_emotion, 0)
        emotion_songs = list(df[df["labels"] == song_label]["uri"].tolist())
        selected_song = random.choice(emotion_songs)
        print(selected_song)
        return jsonify({"uri": selected_song})
        
    except Exception as e:
        print(f"Playlist error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
