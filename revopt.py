import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import streamlit as st
import time as ts 
from PIL import Image
from datetime import time
import pandas as pd


st.set_page_config(layout="wide")


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

names = [
  'nose',
  'left_eye_inner',
  'left_eye',
  'left_eye_outer',
  'right_eye_inner',
  'right_eye',
  'right_eye_outer',
  'left_ear',
  'right_ear',
  'mouth_left',
  'mouth_right',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_pinky',
  'right_pinky',
  'left_index',
  'right_index',
  'left_thumb',
  'right_thumb',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
  'left_heel',
  'right_heel',
  'left_foot_index',
  'right_foot_index',
]

class Landmarks:
    def __init__(self):
        self.landmarkDict = {}
        self.counter = 0
        self.stage = "down"
        self.formType = "bad form"
        self.exercise = "bicep curls"
        self.view = "no view"
    

        
# print(stage, counter)

# POSE PREDICTIONS PRELIMINARY PUNCTIONS

    def getPtCoords(self, lmName):
        if self.landmarkDict:
            point = self.landmarkDict[lmName]
            return point.x, point.y, point.z

    def getPtVis(self, lmName):
        if self.landmarkDict:
            point = self.landmarkDict[lmName]
            return point.visibility
    
    @staticmethod
    def calculate_angle(point1,point2,point3):
        # gotta change but whateva
        x1, y1, z1 = lm.getPtCoords(point1)
        x2, y2, z2 = lm.getPtCoords(point2)
        x3, y3, z3 = lm.getPtCoords(point3)

        radians = np.arctan2(y3-y2, x3-x2) - np.arctan2(y1-y2, x1-x2)
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle 

    # DOING POSES
    def bicepCurlAngles(self):
        left_sew_angle = Landmarks.calculate_angle('left_shoulder', 'left_elbow', 'left_wrist')
        right_sew_angle = Landmarks.calculate_angle('right_shoulder', 'right_elbow', 'right_wrist')
        return left_sew_angle, right_sew_angle


    def checkBicepCurlAngles(self):
        left_sew_angle, right_sew_angle = self.bicepCurlAngles()

        isLeftVis = self.getPtVis('left_thumb') > 0.8
        isRightVis = self.getPtVis('right_thumb') > 0.7

        if isLeftVis and not isRightVis:
            # only check left side angles
            # mark view as leftsideview
            self.view = "left view"
            if (left_sew_angle > 175 or left_sew_angle < 10): # hyperextension 
                    self.formType = "bad form"
            else:
                    self.formType = "good form"
                    if left_sew_angle > 160:
                        self.stage = "down"
                    if left_sew_angle < 30 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1


        elif isRightVis and not isLeftVis:
            # only check right side angles
            # mark view as rightsideview
            self.view = "right view"
            if (right_sew_angle > 175 or right_sew_angle < 10): # hyperextension 
                    self.formType = "bad form"
            else:
                    self.formType = "good form"
                    if right_sew_angle > 160:
                        self.stage = "down"
                    if right_sew_angle < 30 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1

        elif isRightVis and isLeftVis:
            # check both angles
            # mark view as frontview
            self.view = "front view"
        
            if (left_sew_angle > 175 or right_sew_angle > 175 or left_sew_angle < 10 or 
                right_sew_angle < 10): # hyperextension 
                    self.formType = "bad form"
            else:
                    self.formType = "good form"
                    if left_sew_angle > 160 or right_sew_angle > 160:
                        self.stage = "down"
                    if (left_sew_angle < 30 or right_sew_angle < 30) and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1

        else:
            # no view -> hands are not visible 
            self.view = "no view"

#helper function for formatting of video + text
def video_and_text():
    st.markdown ("""
        **Directions:** Position the camera so that your entire body is in view. You
        may need to step back, to see the idenifier pose mapped onto your body in 
        the video. We recommend standing sideways, for more accurate tracking.
        """) 

st.markdown (
    """
    <style>
    [data-testid = "stSidebar"] [aria-expanded = "true"] > div: first-child{
        width: 350px
    }
    [data-testid = "stSidebar"] [aria-expanded = "false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
)
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("FaceMesh SideBar")
st.sidebar.subheader("Parameters")


@st.cache()
def image_resize(image, width=10, height=100, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else: 
        r = width / float(w)
        dim = (width, int(h * r))

  
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


app_mode = st.sidebar.selectbox(
    "Choose the App Mode",
    ("About App", "Live Pose Detection")
)



pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
lm = Landmarks()

def process(image):
    # global counter, stage, formType, view
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

# Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            lm.landmarkDict[names[i]] = landmark
    
    if lm.exercise == "bicep curls":  
        lm.checkBicepCurlAngles()
    # print(lm.bicepCurlAngles())
    # print(getPtVis('left_thumb'), getPtVis('right_thumb'))
    print("#####-----FEEDBACK-----######")
    if lm.view == "no view":
        print("Please center yourself in the frame")
    else:
        print("View: ", lm.view)
        print("Form: ", lm.formType)
        print("Counter: ", lm.counter)
        print("Stage: ", lm.stage)
    print("-----------------------------")

    return cv2.flip(image, 1)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

if app_mode == 'About App':
     
    st.markdown('''**In this app, you will be able to accuratly analyze your form
    in real-time whilst getting expert feedback on the correctness, count of reps, 
    and expert tips.**''')

    st.markdown ('''This app was built using Streamlit and Mediapipe, to analyse 
    and present live pose recognition results.''')


st.markdown (
    """
    <style>
    [data-testid = "stSidebar"] [aria-expanded = "true"] > div: first-child{
        width: 350px
    }
    [data-testid = "stSidebar"] [aria-expanded = "false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    )
    </style>
    """,
    unsafe_allow_html=True,
)


if app_mode == 'Live Pose Detection':
    st.title("Live Pose Detection")
    st.set_option("deprecation.showfileUploaderEncoding", False)
    

    st.markdown (
        """
        <style>
        [data-testid = "stSidebar"] [aria-expanded = "true"] > div: first-child{
            width: 350px
        }
        [data-testid = "stSidebar"] [aria-expanded = "false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
    )
        </style>
        """,
        unsafe_allow_html=True,
    )

    select_pose_type = st.sidebar.selectbox("Select Pose Type", ("Bicep Curl", "Lunges", "Squats", "Glute Brigde"))
    if select_pose_type == "Bicep Curl":
        #Display Bicep Pose on the screen above the video
        st.header("Bicep Curl Pose")
        video_and_text()

    elif select_pose_type == "Lunges":
        #Display Lunges Pose on the screen above the video
        st.header("Lunge Pose")
        video_and_text()

    elif select_pose_type == "Squats":
        #Display Squats Pose on the screen above the video
        st.header("Squat Pose")
        video_and_text()

    elif select_pose_type == "Glute Brigde":
        #Display Glute Brigde Pose on the screen above the video
        st.header("Glute Brigde Pose")
        video_and_text()

    st.markdown (
        """
        <style>
        [data-testid = "stSidebar"] [aria-expanded = "true"] > div: first-child{
            width: 350px
        }
        [data-testid = "stSidebar"] [aria-expanded = "false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
    )
        </style>
        """,
        unsafe_allow_html=True,
    )

    #Changing picture in response to good form or your form sucks...

    # loopCounter = 0
    # while True:
    #     placeholder = st.empty()
    #     with placeholder.container():
    #         col1, col2 = st.columns([1,3])
    #         col1.subheader("Feedback:")
    #         print(lm.counter, lm.stage, lm.formType, lm.view)

            # if lm.formType == "good form":
            #     good_form_smile= Image.open('good.jpg')
            #     col1.image(good_form_smile)
            #     col1.markdown("<h2 style='text-align: center; color: white;'>Great Form! Keep it up! </h2>", unsafe_allow_html=True)

            
            # else:
            #     bad_form_smile= Image.open('bad.jpg')
            #     col1.image(bad_form_smile)
            #     col1.markdown("<h2 style='text-align: center; color: white;'>Oh no! This is bad form! You can do better! </h2>", unsafe_allow_html=True)


            # reps = st.sidebar.slider("Reps:", 0, 2, key=f"reps-slider-{loopCounter}")
            # number_of_reps = 0
            # sets = st.sidebar.slider("Sets:", 0, 20, key=f"sets-slider-{loopCounter}")


            # if reps == lm.counter: 
            #     number_of_reps += 1
            #     col1.markdown(f"<h2 style='text-align: center; color: white;'> {number_of_reps} Reps Complete!</h2>", unsafe_allow_html=True)

            # if number_of_reps == sets:
            #     col1.markdown(f"<h2 style='text-align: center; color: white;'> Workout Complete!</h2>", unsafe_allow_html=True)
                
            # col1.markdown(f"<h1 style='text-align: center; color: white;'>{lm.counter} </h1>", unsafe_allow_html=True)

            # show angles 
            # st.sidebar.checkbox("Show Angles", key=f"angle-checkbox-{loopCounter}")

        #     if "Show Angles": 
        #         st.text("weee")
        #         #something to show angles :) 

        #     #timer with progress bar

        #     def converter(value):
        #         mins, secs, milisecs = value.split(":")
        #         seconds = int(mins) * 60 + int(secs)
        #         return seconds


        #     val = st.time_input("Timer", value = time(0, 0, 0), key=f"timer-{loopCounter}" )
        #     # print (val)

        #     st.write(f"Timer is set to {val}")
        #     sec = converter(str(val))
        #     bar = st.progress(0)
        #     per = sec/100
        #     progress_status = st.empty()
        #     for i in range(100):
        #         bar.progress((i +1))
        #         progress_status.write(str(i+1) + "%")
        #         ts.sleep(per)
        # loopCounter += 1
        # placeholder.empty()