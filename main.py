import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import numpy as np
import time

# Carregamento do modelo YOLOv8s
model = YOLO("yolov8s.pt") 

st.title("Projeto Visão Computacional - Detecção de Objetos com YOLO e OpenCV")

st.write("Grupo 7: Gabriel, Joyce, Yan")

if "running" not in st.session_state:
    st.session_state.running = False

# Sessão para seleção do tipo de input: Webcam ou Videos nos formatos: MP4, avi ou mov
source_type = st.radio("Video Input: ", ["Upload", "Webcam"])

video_path = None

if source_type == "Upload":
    video_file = st.file_uploader("Upload your video here", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

elif source_type == "Webcam":
    video_path = 0

col1, col2 = st.columns(2)
if col1.button("Start Detection"):
    st.session_state.running = True
if col2.button("Stop Detection"):
    st.session_state.running = False

stframe = st.empty()

if st.session_state.running:
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 0.03

        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, stream=True)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

            time.sleep(delay)

        cap.release()
    else:
        st.warning("Upload a Video or Use your Webcam.")
