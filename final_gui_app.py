import os
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pyjokes
import smtplib
from email.mime.text import MIMEText
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ---------------------------
# CONFIG
# ---------------------------
SENDER_EMAIL = "shinytasleem1905@gmail.com"
APP_PASSWORD = "&hinytasleem@_1905"
GUARDIAN_EMAIL = "shinytasleem1905@gmail.com"

SPOTIFY_CLIENT_ID = "70e78505069c49939f27134ec8b50df1"
SPOTIFY_CLIENT_SECRET = "172a76236be3450097a8b818c7dc30bf"

# ---------------------------
# Models
# ---------------------------
EMOTION_MODEL_PATH = "emotion_model.h5"
AGE_MODEL_PATH = "age_model.h5"
SMILE_MODEL_PATH = "smile_mobilenetv2.h5"
MASK_MODEL_PATH = "mask_model.h5"   # NOT AVAILABLE

print("Loading models...")

emotion_model = load_model(EMOTION_MODEL_PATH) if os.path.exists(EMOTION_MODEL_PATH) else None

if os.path.exists(AGE_MODEL_PATH):
    try:
        age_model = load_model(
            AGE_MODEL_PATH,
            compile=False,
            custom_objects={"mse": lambda y_true, y_pred: y_pred}
        )
    except:
        print("⚠ Age model failed.")
        age_model = None
else:
    age_model = None

smile_model = load_model(SMILE_MODEL_PATH) if os.path.exists(SMILE_MODEL_PATH) else None
mask_model = load_model(MASK_MODEL_PATH) if os.path.exists(MASK_MODEL_PATH) else None

print("Models loaded.\n")

# ---------------------------
# Helpers
# ---------------------------
def send_alert_email():
    try:
        msg = MIMEText("⚠️ ALERT: Fear emotion detected.")
        msg["Subject"] = "Emergency – Fear Detected"
        msg["From"] = SENDER_EMAIL
        msg["To"] = GUARDIAN_EMAIL

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, GUARDIAN_EMAIL, msg.as_string())

        return "⚠️ FEAR DETECTED!\nEmail Alert Sent!"
    except Exception as e:
        return f"Email Error: {str(e)}"


def suggest_songs_spotify():
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))
        res = sp.search(q="calming music", type="track", limit=5)
        songs = [f"{t['name']} - {t['artists'][0]['name']}" for t in res['tracks']['items']]
        return "🎵 Calm Songs:\n" + "\n".join(songs)
    except Exception as e:
        return f"Spotify Error: {str(e)}"

def get_joke():
    try:
        return "😂 Joke:\n" + pyjokes.get_joke()
    except:
        return "😂 Here's a smile!"

# ---------------------------
# ★★★ FIXED MASK DETECTION ★★★
# ---------------------------
def heuristic_mask_check(face_bgr):
    face = cv2.resize(face_bgr, (200, 200))

    h = face.shape[0]
    upper = face[0:int(h*0.45), :]
    lower = face[int(h*0.45):int(h*0.95), :]

    # Convert to HSV
    hsv_lower = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)

    # Updated skin ranges (more robust)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Lower face skin detection
    mask = cv2.inRange(hsv_lower, lower_skin, upper_skin)

    skin_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    skin_ratio = skin_pixels / (total_pixels + 1e-6)

    # ⭐ More realistic thresholds
    if skin_ratio < 0.18:
        return "Mask"
    elif skin_ratio > 0.30:
        return "No Mask"

    # Uncertain zone → check upper half also
    hsv_upper = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    mask_u = cv2.inRange(hsv_upper, lower_skin, upper_skin)
    upper_ratio = cv2.countNonZero(mask_u) / (mask_u.size + 1e-6)

    if upper_ratio > 0.35:
        return "No Mask"

    return "Mask"

# ---------------------------
# Age ranges
# ---------------------------
def age_to_range(age_val):
    try:
        age = int(age_val)
    except:
        return "--"
    if age < 13: return "0-12"
    if age < 18: return "13-17"
    if age < 25: return "18-24"
    if age < 35: return "25-34"
    if age < 50: return "35-49"
    return "50+"

# ---------------------------
# Face Detection
# ---------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------
# GUI
# ---------------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Personal Care Assistant")
app.geometry("920x780")

title_label = ctk.CTkLabel(app, text="Personal Care Assistant",
                           font=("Arial", 32, "bold"),
                           text_color="#4a4aff")
title_label.pack(pady=10)

camera_frame = ctk.CTkFrame(app, width=640, height=380)
camera_frame.pack()
camera_label = ctk.CTkLabel(camera_frame, text="")
camera_label.pack()

info_frame = ctk.CTkFrame(app)
info_frame.pack(pady=10)

age_box = ctk.CTkLabel(info_frame, text="AGE\n--", width=260, height=100,
                       fg_color="#e8e8ff", corner_radius=15, font=("Arial", 20))
age_box.grid(row=0, column=0, padx=20, pady=10)

smile_box = ctk.CTkLabel(info_frame, text="SMILE STATUS\n--", width=260, height=100,
                         fg_color="#e8e8ff", corner_radius=15, font=("Arial", 20))
smile_box.grid(row=0, column=1, padx=20, pady=10)

mask_box = ctk.CTkLabel(info_frame, text="MASK STATUS\n--", width=260, height=100,
                        fg_color="#e8e8ff", corner_radius=15, font=("Arial", 20))
mask_box.grid(row=1, column=0, padx=20, pady=10)

emotion_box = ctk.CTkLabel(info_frame, text="DETECTED EMOTION\n--", width=260, height=100,
                           fg_color="#e8e8ff", corner_radius=15, font=("Arial", 20))
emotion_box.grid(row=1, column=1, padx=20, pady=10)

# ACTION BOX
action_box = ctk.CTkLabel(app, text="ACTIONS\n--", width=650, height=150,
                          fg_color="#dcdcff", corner_radius=15,
                          font=("Arial", 18), justify="left")
action_box.pack(pady=10)

status_bar = ctk.CTkLabel(app, text="Press X to close.", anchor="w")
status_bar.pack(fill="x", padx=10, pady=5)

# ---------------------------
# Webcam loop
# ---------------------------
cap = cv2.VideoCapture(0)
FRAME_SKIP = 4
frame_count = 0

last_emotion = None
cooldown = 40

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def handle_emotion_action(emotion):
    if emotion == "Sad":
        action_box.configure(text=f"ACTIONS\n{get_joke()}")
    elif emotion == "Angry":
        action_box.configure(text=f"ACTIONS\n{suggest_songs_spotify()}")
    elif emotion == "Fear":
        action_box.configure(text=f"ACTIONS\n{send_alert_email()}")

def update_frame():
    global frame_count, last_emotion, cooldown

    ret, frame = cap.read()
    if not ret:
        app.after(10, update_frame)
        return

    # show camera
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_label.imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
    camera_label.configure(image=camera_label.imgtk)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    frame_count += 1
    if cooldown > 0:
        cooldown -= 1

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]

        if frame_count % FRAME_SKIP == 0:
            # Emotion
            try:
                e = cv2.resize(face, (48, 48))
                e = cv2.cvtColor(e, cv2.COLOR_BGR2GRAY)
                e = e.astype("float32")/255.0
                e = np.expand_dims(e, axis=(0, -1))
                pred = emotion_model.predict(e, verbose=0)[0]
                emotion = emotion_labels[np.argmax(pred)]
            except:
                emotion = "--"

            # Smile
            try:
                s = cv2.resize(face, (96, 96))
                s = preprocess_input(s)
                sm = smile_model.predict(np.expand_dims(s, 0), verbose=0)[0][0]
                smile = "Smiling" if sm > 0.5 else "Not Smiling"
            except:
                smile = "--"

            # Age
            try:
                a = cv2.resize(face, (128, 128)).astype("float32")/255.0
                pred_age = age_model.predict(np.expand_dims(a, 0), verbose=0)[0][0]
                age = age_to_range(pred_age)
            except:
                age = "--"

            # MASK (improved)
            try:
                mask = heuristic_mask_check(face)
            except:
                mask = "--"

            # UPDATE GUI
            age_box.configure(text=f"AGE\n{age}")
            smile_box.configure(text=f"SMILE STATUS\n{smile}")
            mask_box.configure(text=f"MASK STATUS\n{mask}")
            emotion_box.configure(text=f"DETECTED EMOTION\n{emotion}")

            # ACTION
            if emotion != last_emotion and emotion != "--":
                last_emotion = emotion
                cooldown = 40
                handle_emotion_action(emotion)

    app.after(10, update_frame)

update_frame()

def on_close():
    try:
        cap.release()
    except:
        pass
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)
app.mainloop()
