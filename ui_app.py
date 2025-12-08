import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Detecting Assistant")
app.geometry("900x700")

title_label = ctk.CTkLabel(app, text="Detecting Assistant",
                           font=("Arial", 32, "bold"),
                           text_color="#4a4aff")
title_label.pack(pady=15)

# -----------------------------
# Frame for camera preview
# -----------------------------
camera_frame = ctk.CTkFrame(app, width=500, height=350, corner_radius=20)
camera_frame.pack(pady=10)

camera_label = ctk.CTkLabel(camera_frame, text="")
camera_label.pack()

# -----------------------------
# Info Grid (Age, Smile, Emotion, Mask)
# -----------------------------
info_frame = ctk.CTkFrame(app, corner_radius=20)
info_frame.pack(pady=20)

age_box = ctk.CTkLabel(info_frame, text="AGE\n--", width=200, height=80,
                       fg_color="#e8e8ff", corner_radius=15,
                       font=("Arial", 18, "bold"))
age_box.grid(row=0, column=0, padx=20, pady=10)

smile_box = ctk.CTkLabel(info_frame, text="SMILE STATUS\n--", width=200, height=80,
                         fg_color="#e8e8ff", corner_radius=15,
                         font=("Arial", 18, "bold"))
smile_box.grid(row=0, column=1, padx=20, pady=10)

mask_box = ctk.CTkLabel(info_frame, text="MASK STATUS\n--", width=200, height=80,
                        fg_color="#e8e8ff", corner_radius=15,
                        font=("Arial", 18, "bold"))
mask_box.grid(row=1, column=0, padx=20, pady=10)

emotion_box = ctk.CTkLabel(info_frame, text="DETECTED EMOTION\n--", width=200, height=80,
                           fg_color="#e8e8ff", corner_radius=15,
                           font=("Arial", 18, "bold"))
emotion_box.grid(row=1, column=1, padx=20, pady=10)

# -----------------------------
# CAMERA UPDATE LOOP
# -----------------------------
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    app.after(10, update_frame)

update_frame()

app.mainloop()
