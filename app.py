import streamlit as st
import torch
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

model = load_model()
mp_face_mesh = mp.solutions.face_mesh

st.title("📸 Cheating Detection System")
st.write("Upload an image of students in an exam room. The system will detect suspicious behavior or phone usage.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # 🔍 Detect phones using YOLOv5
    results = model(rgb)
    detections = results.pandas().xyxy[0]

    for i, row in detections.iterrows():
        if row['name'] == 'cell phone' and row['confidence'] > 0.4:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, 'Cheating (Phone)', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 🧠 Detect suspicious head turns using Mediapipe
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results_mp = face_mesh.process(rgb)

    if results_mp.multi_face_landmarks:
        for face_landmarks in results_mp.multi_face_landmarks:
            lm = face_landmarks.landmark
            try:
                image_points = np.array([
                    (lm[1].x * w, lm[1].y * h), (lm[33].x * w, lm[33].y * h),
                    (lm[263].x * w, lm[263].y * h), (lm[61].x * w, lm[61].y * h),
                    (lm[291].x * w, lm[291].y * h), (lm[199].x * w, lm[199].y * h)
                ], dtype='double')

                model_points = np.array([
                    (0, 0, 0), (-30, -30, -30), (30, -30, -30),
                    (-40, 40, -30), (40, 40, -30), (0, 75, -50)
                ])

                camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype='double')
                dist = np.zeros((4,1))
                _, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist)
                rotM = cv2.Rodrigues(rvec)[0]
                proj = np.hstack((rotM, tvec))
                yaw = cv2.decomposeProjectionMatrix(proj)[6][1][0]

                cx, cy = int(lm[1].x * w), int(lm[1].y * h)
                try:
    color = (0, 0, 255) if abs(yaw) > 15 else (0, 255, 0)
finally:
    print("Finished calculating color.")
