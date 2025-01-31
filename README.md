# Human Pose Estimation using Machine Learning  
![Demo](Sample%20Files/Demo.gif)  

A real-time human pose estimation system that processes **images**, **videos**, and **webcam feeds** using OpenCV DNN and MediaPipe.  

## Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Author](#author)  

---

## Overview  
This project estimates human poses using a hybrid approach:  
- **OpenCV DNN** for images and pre-recorded videos.  
- **MediaPipe** for real-time webcam processing.  
- **Streamlit** for an interactive user interface.  

Ideal for applications in sports analysis, healthcare monitoring, and surveillance.  

---

## Features  
- ✅ Multi-input support (Image/Video/Webcam).  
- ✅ Adjustable confidence threshold and frame skip.  
- ✅ Real-time processing with optimization for edge devices.  
- ✅ Customizable background image with transparency.  

---

## Installation  

### Prerequisites  
- Python 3.8 - 3.12  
- pip  

### Steps  
1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/0monish/Human-Pose-Estimation-using-ML.git  
   cd Human-Pose-Estimation-using-ML  
   ```  
2. **Install Dependencies**  
   ```bash  
   pip install opencv-python==4.5.5.64 streamlit==1.13.0 numpy==1.23.5 mediapipe==0.8.11 Pillow==9.4.0  
   ``` 
---

## Usage  

### Launch the App  
```bash  
streamlit run human_pose_estimation.py  
```  

### Select Input Type  
- **Image:** Upload a JPG/PNG/etc file.  
- **Video:** Upload an MP4/MOV/etc file and adjust frame skip/resolution.  
- **Webcam:** Start live pose estimation.  

### Adjust Parameters  
- **Confidence Threshold:** Lower values detect more joints (may include noise).  
- **Frame Skip:** Process every 2nd/3rd/4th frame for faster video analysis.  
- **Processing Width:** Reduce for faster inference (e.g., 320px).  

---

## Author  
Monish Khandelwal
