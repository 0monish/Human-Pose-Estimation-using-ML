import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import mediapipe as mp


# INITIALIZE THE MODEL
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

width = 368
height = 368
thres = 0.2

# MEDIAPIPE POSE SETUP
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ORIGINAL IMAGE PROCESSING FUNCTION (NO MODIFICATIONS)
def poseDetector(frame, threshold):
    h, w = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height), 
                               (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int((w * point[0]) / out.shape[3])
        y = int((h * point[1]) / out.shape[2])
        points.append((x, y) if conf > threshold else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# OPTIMIZED VIDEO PROCESSING FUNCTION WITH FRAME CONTROLS
def process_video(uploaded_file, threshold, frame_skip, process_w):
    # VIDEO-SPECIFIC OPTIMIZATIONS
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name

    cap = cv2.VideoCapture(input_path)
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # VIDEO-SPECIFIC PROCESSING WITH OPTIMIZATIONS
        h, w = frame.shape[:2]
        processed_frame = cv2.resize(frame, (process_w, int(process_w*h/w)))
        
        # PROCESS FRAME WITH POSE DETECTION (ASSUMING NET IS PREDEFINED AND READY TO USE)
        blob = cv2.dnn.blobFromImage(processed_frame, 1.0, (width, height), 
                                   (127.5, 127.5, 127.5), swapRB=True, crop=False)
        net.setInput(blob)
        out = net.forward()
        out = out[:, :19, :, :]

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = int((w * point[0]) / out.shape[3])
            y = int((h * point[1]) / out.shape[2])
            points.append((x, y) if conf > threshold else None)

        for pair in POSE_PAIRS:
            partFrom, partTo = pair
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, use_container_width=True)
        
        processed_frames += 1
        progress_bar.progress(min(processed_frames / (total_frames // frame_skip), 1.0))

    cap.release()
    return True

# FUNCTION FOR LIVE WEBCAM POSE ESTIMATION
def process_webcam_mediapipe(threshold):
    cap = cv2.VideoCapture(0) 
    
    # SET VIDEO RESOLUTION FOR WEBCAM (OPTIONAL, ADJUST AS NEEDED FOR YOUR LAPTOP)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # INITIALIZE MEDIAPIPE POSE
    with mp_pose.Pose(static_image_mode=False, model_complexity=2, 
                      enable_segmentation=False, min_detection_confidence=threshold, 
                      min_tracking_confidence=threshold) as pose:
        
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # FLIP FRAME HORIZONTALLY FOR A MIRROR-LIKE EFFECT (OPTIONAL)
            frame = cv2.flip(frame, 1)

            # CONVERT THE BGR IMAGE TO RGB FOR MEDIAPIPE
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # PROCESS THE FRAME WITH MEDIAPIPE POSE
            results = pose.process(frame_rgb)

            # DRAW LANDMARKS ON THE FRAME
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )   

            # CONVERT THE BGR FRAME BACK TO RGB FOR STREAMLIT
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # DISPLAY THE PROCESSED FRAME IN STREAMLIT
            frame_placeholder.image(rgb_frame, caption="Webcam Pose Estimation", use_container_width=True)

        cap.release()

# STREAMLIT UI
st.title("Human Pose Estimation using Machine Learning")
option = st.sidebar.selectbox("Choose Input Type", ["Image", "Video", "Webcam"])
thres = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.05)

if option == "Image":
    st.subheader("Image Pose Estimation")
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        image = Image.open(img_file)
        frame = np.array(image)
        
        st.subheader("Original Image")
        st.image(frame, use_container_width=True)
        
        processed_frame = poseDetector(frame, thres)
        st.subheader("Processed Image")
        st.image(processed_frame, use_container_width=True)

elif option == "Video":
    st.subheader("Video Pose Estimation")
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov"])
    frame_skip = st.sidebar.slider("Frame Skip", 1, 4, 2, 
                                   help="Process every Nth frame (higher = faster)")
    process_w = st.sidebar.slider("Processing Width", 320, 1280, 640, 160,
                                  help="Lower values process faster")
    
    if video_file is not None:
        st.subheader("Original Video")
        st.video(video_file)
        
        if st.button("Process Video"):
            with st.spinner("Processing..."):
                process_video(video_file, thres, frame_skip, process_w)

elif option == "Webcam":
    st.subheader("Webcam Pose Estimation")
    if st.button("Start Webcam Processing"):
        with st.spinner("Processing webcam..."):
            process_webcam_mediapipe(thres)
