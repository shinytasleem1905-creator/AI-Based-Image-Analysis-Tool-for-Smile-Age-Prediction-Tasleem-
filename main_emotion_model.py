import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyjokes
import smtplib
from email.mime.text import MIMEText
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# -----------------------------------------
# Load Models
# -----------------------------------------
emotion_model = load_model("emotion_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# OPTIONAL — only if you have trained mask model
# mask_model = load_model("mask_detector.h5")

# OPTIONAL — only if you have trained age model
# age_model = load_model("age_model.h5")


# -----------------------------------------
# Emotion Labels (FER-2013)
# -----------------------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# -----------------------------------------
# Email Alert Function (For Fear)
# -----------------------------------------
def send_alert_email():
    sender = "shinytasleem1905@gmail.com"
    password = "&hinytasleem19"          # Gmail App Password
    receiver = "shinytasleem1905@gmail.com"

    msg = MIMEText("⚠️ ALERT: A fear emotion is detected. Please check immediately.")
    msg["Subject"] = "EMERGENCY – Fear Detected"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print("Alert email sent successfully!")
    except Exception as e:
        print("Error sending email:", e)


# -----------------------------------------
# Spotify Song Suggestions (For Anger)
# -----------------------------------------
def suggest_songs():
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id="5ea663f4739442a988fe6dbc7dcc2a6b",
            client_secret="f5f952670bed434fbbed139e2de1560c"
        ))

        results = sp.search(q="calming music", type="track", limit=5)

        print("\n🎵 Calm Song Suggestions:\n")
        for i, track in enumerate(results["tracks"]["items"]):
            print(f"{i+1}. {track['name']} - {track['artists'][0]['name']}")
        print()

    except Exception as e:
        print("Spotify error:", e)


# -----------------------------------------
# Mask Detection Placeholder
# -----------------------------------------
def detect_mask(face_img):
    # If you have a real model, replace this line:
    return "Mask Not Detected"


# -----------------------------------------
# Cool-down system (to avoid infinite actions)
# -----------------------------------------
last_emotion = None
cooldown_frames = 30
cooldown_counter = 0


# -----------------------------------------
# Start Webcam
# -----------------------------------------
cap = cv2.VideoCapture(0)
print("Starting Emotion Detection...\nPress Q to Quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if cooldown_counter > 0:
        cooldown_counter -= 1

    for (x, y, w, h) in faces:

        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Preprocess face
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        roi_gray_norm = roi_gray_resized.astype("float") / 255.0
        roi_gray_norm = np.expand_dims(roi_gray_norm, axis=-1)
        roi_gray_norm = np.expand_dims(roi_gray_norm, axis=0)

        # Predict emotion
        prediction = emotion_model.predict(roi_gray_norm, verbose=0)[0]
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        # Show emotion on screen
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Only trigger action when emotion changes
        if cooldown_counter == 0 and emotion != last_emotion:
            last_emotion = emotion
            cooldown_counter = cooldown_frames

            if emotion == "Fear":
                send_alert_email()

            elif emotion == "Sad":
                print("\n😂 Joke:", pyjokes.get_joke(), "\n")

            elif emotion == "Angry":
                suggest_songs()

        # Mask detection
        mask_status = detect_mask(roi_gray_resized)
        cv2.putText(frame, mask_status, (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Emotion + Mask Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
