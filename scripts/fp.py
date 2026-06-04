import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "models/exercise_classifier.pkl"
)

encoder = joblib.load(
    "models/label_encoder.pkl"
)

# =========================
# MEDIAPIPE
# =========================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

# =========================
# WE NEED SAME COLUMN ORDER
# =========================

training_columns = pd.read_csv(
    "data/merged/landmarks.csv",
    nrows=1
).drop(
    columns=["vid_id", "frame_order"]
).columns.tolist()

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = pose.process(rgb)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks.landmark

        row = {}

        names = [
            "nose",
            "left_eye_inner","left_eye","left_eye_outer",
            "right_eye_inner","right_eye","right_eye_outer",
            "left_ear","right_ear",
            "mouth_left","mouth_right",
            "left_shoulder","right_shoulder",
            "left_elbow","right_elbow",
            "left_wrist","right_wrist",
            "left_pinky_1","right_pinky_1",
            "left_index_1","right_index_1",
            "left_thumb_2","right_thumb_2",
            "left_hip","right_hip",
            "left_knee","right_knee",
            "left_ankle","right_ankle",
            "left_heel","right_heel",
            "left_foot_index","right_foot_index"
        ]

        for i, name in enumerate(names):

            row[f"x_{name}"] = landmarks[i].x
            row[f"y_{name}"] = landmarks[i].y
            row[f"z_{name}"] = landmarks[i].z

        sample = pd.DataFrame([row])

        sample = sample[
            training_columns
        ]

        pred = model.predict(sample)

        exercise = encoder.inverse_transform(
            pred
        )[0]

        cv2.putText(
            frame,
            exercise,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow(
        "Exercise Classifier",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()