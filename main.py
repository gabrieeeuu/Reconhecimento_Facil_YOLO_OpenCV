import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import numpy as np
import time

from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import av

if "filter" not in st.session_state:
  st.session_state.filter = "None"

canny_threshold1 = 100
canny_threshold2 = 200

blur_threshold1 = 11
blur_threshold2 = 21

@st.cache_data
def transform(img):
  match st.session_state.filter:
    case "Blur":
      img = cv2.GaussianBlur(img, (blur_threshold1, blur_threshold2), 0)
    case "Canny":
      img = cv2.cvtColor(cv2.Canny(img, canny_threshold1, canny_threshold2), cv2.COLOR_GRAY2BGR)
    case "Grayscale":
      img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    case "Sepia":
      kernel = np.array(
          [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
      )
      img = cv2.transform(img, kernel)
    case "Invert":
      img = cv2.bitwise_not(img)
    case "None":
      pass

  return img

# Carregamento do modelo YOLOv8s
model = YOLO("yolov8s.pt") 

st.title("Projeto Visão Computacional - Detecção de Objetos com YOLO e OpenCV")

st.write("Grupo 7: Gabriel, Joyce, Yan")

if "running" not in st.session_state:
  st.session_state.running = False

# Sessão para seleção do tipo de input: Webcam ou Videos nos formatos: MP4, avi ou mov
source_type = st.radio("Video Input: ", ["Webcam", "Upload"])

video_path = None

if source_type == "Upload":
  video_file = st.file_uploader("Upload your video here", type=["mp4", "avi", "mov"])
  if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

elif source_type == "Webcam":
  video_path = 0

col1, col2 = st.columns(2, gap="large")
if col1.button("Start Detection"):
  st.session_state.running = True
if col2.button("Stop Detection"):
  st.session_state.running = False

filters = ["None", "Blur", "Grayscale", "Sepia", "Canny", "Invert"]
  
st.session_state.filter = st.radio("Aplique filtros: ", filters, index=None, horizontal=True)

stframe = st.empty()

if video_path is not None:
    
  cap = cv2.VideoCapture(video_path)

  fps = cap.get(cv2.CAP_PROP_FPS)
  delay = 1 / fps if fps > 0 else 0.03

  while st.session_state.running and cap.isOpened():
    ret, frame = cap.read()
    
    frame = transform(frame)
    
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

    if st.session_state.filter != "Grayscale": frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame, channels="RGB")

    time.sleep(delay)

  cap.release()
else:
  st.warning("Upload a Video or Use your Webcam.")
