import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pyttsx3
import time
import threading
import queue


# ===============================
# Load model and encoder
# ===============================

model = load_model(
    "bicep_model/exercise_model.h5"
)

encoder = pickle.load(
    open(
        "bicep_model/encoder.pkl",
        "rb"
    )
)

speech_queue = queue.Queue()

def speech_worker():

    while True:

        text = speech_queue.get()


        engine.say(text)
        engine.runAndWait()

        speech_queue.task_done()

#initialise text to speech engine

engine = pyttsx3.init()

engine.setProperty(
    "rate",
    150
)

speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()


# ===============================
# MediaPipe setup
# ===============================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(
            c[1] - b[1],
            c[0] - b[0]
        )
        -
        np.arctan2(
            a[1] - b[1],
            a[0] - b[0]
        )
    )

    angle = np.abs(
        radians * 180.0 / np.pi
    )

    if angle > 180:
        angle = 360 - angle

    return angle

# ===============================
# Webcam
# ===============================

cap = cv2.VideoCapture(0)

sequence = []

prediction_history = []
stable_prediction = "No exercise"

rep_count = 0
stage = None
curl_angle = 0

start_elbow_x = None
elbow_drift = 0
feedback_list = []
last_spoken_feedback = ""

active_errors = set()

last_speech_time = 0
min_curl_angle = 180

error_counts = {
    "Keep Elbows Close To Torso": 0,
    "Complete Full Range": 0
}
elbow_drift_detected = False

while True:
    

    ret, frame = cap.read()

    if not ret:
        break

    # mirror webcam
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = pose.process(rgb)

    if result.pose_landmarks:

        # draw skeleton

        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks=[]
        feedback_list = []
        elbow_drift = 0

        for lm in result.pose_landmarks.landmark:

            landmarks.extend(
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
            )

        # add current frame

        sequence.append(
            landmarks
        )

        # keep only last 30 frames

        sequence = sequence[-30:]


        # ===============================
        # Motion detection
        # ===============================

        if len(sequence)>=2:

            movement=np.mean(
                np.abs(
                    np.array(sequence[-1]) -
                    np.array(sequence[-2])
                )
            )

            # almost no movement

            if movement<0.01:

                cv2.putText(
                    frame,
                    "No exercise",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Movement:{movement:.4f}",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )

                cv2.imshow(
                    "Exercise Recognition",
                    frame
                )

                key=cv2.waitKey(1)

                if key==27:
                    break

                continue


        # ===============================
        # LSTM prediction
        # ===============================

        if len(sequence)==30:

            sample=np.expand_dims(
                np.array(sequence),
                axis=0
            )

            prediction=model.predict(
                sample,
                verbose=0
            )

            confidence=np.max(
                prediction
            )

            pred=np.argmax(
                prediction
            )


            if confidence >= 0.85:

                exercise = encoder.inverse_transform(
                    [pred]
                )[0]

                prediction_history.append(
                    exercise
                )

                prediction_history = (
                    prediction_history[-10:]
                )

                if len(prediction_history) == 10:

                    stable_prediction = max(
                        set(prediction_history),
                        key=prediction_history.count
                    )

                else:

                    stable_prediction = exercise

                if stable_prediction == "bicep_curl":

                    lm = result.pose_landmarks.landmark

                    shoulder = [
                        lm[
                            mp_pose.PoseLandmark.LEFT_SHOULDER.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.LEFT_SHOULDER.value
                        ].y
                    ]

                    elbow = [
                        lm[
                            mp_pose.PoseLandmark.LEFT_ELBOW.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.LEFT_ELBOW.value
                        ].y
                    ]

                    wrist = [
                        lm[
                            mp_pose.PoseLandmark.LEFT_WRIST.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.LEFT_WRIST.value
                        ].y
                    ]

                    curl_angle = calculate_angle(
                        shoulder,
                        elbow,
                        wrist
                    )

                    min_curl_angle = min(
                        min_curl_angle,
                        curl_angle
                    )
                    
                    # ==========================
                    # ELBOW DRIFT DETECTION
                    # ==========================

                    if start_elbow_x is not None:

                        elbow_drift = abs(
                            elbow[0] - start_elbow_x
                        )

                        if elbow_drift > 0.05:

                            feedback_list.append(
                                "Keep Elbows Close To Torso"
                            )

                            elbow_drift_detected = True

                    # ==========================
                    # REP COUNTING
                    # ==========================

                    if curl_angle > 150:

                        stage = "DOWN"

                        if start_elbow_x is None:
                            start_elbow_x = elbow[0]

                    elif (
                        curl_angle < 50
                        and stage == "DOWN"
                    ):

                        # =====================
                        # FULL RANGE CHECK
                        # =====================

                        if min_curl_angle > 60:

                            feedback_list.append(
                                "Complete Full Range"
                            )

                            error_counts[
                                "Complete Full Range"
                            ] += 1

                        stage = "UP"

                        if elbow_drift_detected:

                            error_counts[
                                "Keep Elbows Close To Torso"
                            ] += 1

                            elbow_drift_detected = False

                        rep_count += 1

                        min_curl_angle = 180

                        print(
                            f"Bicep Curl Rep Count: {rep_count}"
                        )


            cv2.putText(
                frame,
                f"{stable_prediction}: {confidence:.2f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"Angle: {curl_angle:.1f}",
                (20,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            cv2.putText(
                    frame,
                    f"MinAngle:{min_curl_angle:.1f}",
                    (20,280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )
            
            cv2.putText(
                frame,
                f"Reps: {rep_count}",
                (20,160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,0,0),
                2
            )

            cv2.putText(
                frame,
                f"Drift:{elbow_drift:.3f}",
                (20,200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )
            
            current_errors = set(feedback_list)

            new_errors = current_errors - active_errors

            for error in new_errors:
                speech_queue.put(error)

            active_errors = current_errors
            
            for i, feedback in enumerate(feedback_list):

                cv2.putText(
                    frame,
                    feedback,
                    (20, 320 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


            cv2.putText(
                frame,
                f"Movement:{movement:.4f}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

    else:

        cv2.putText(
            frame,
            "Body not detected",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    cv2.putText(
        frame,
        "Press ESC to Exit",
        (20,420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow(
        "Exercise Recognition",
        frame
    )

    key=cv2.waitKey(1)

    if key == 27:

        print("\n========== WORKOUT SUMMARY ==========")

        print(
            f"Total Reps: {rep_count}"
        )

        for error, count in error_counts.items():

            print(
                f"{error}: {count}"
            )

        print("=====================================\n")

        break

speech_queue.put(None)
speech_thread.join(timeout=1)

cap.release()
cv2.destroyAllWindows()