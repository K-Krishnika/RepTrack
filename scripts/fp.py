import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import queue
import threading

from tensorflow.keras.models import load_model
import pyttsx3


# ===============================
# Load Model and Encoder
# ===============================

model = load_model("bicep_model/exercise_model.h5")
encoder = pickle.load(open("bicep_model/encoder.pkl", "rb"))


# ===============================
# Text-to-Speech Setup
# ===============================

engine = pyttsx3.init()
engine.setProperty("rate", 150)

speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Tracks last time each feedback was spoken (feedback -> timestamp)
last_spoken_times = {}
SPEECH_COOLDOWN = 5.0  # seconds


def speak_feedback(feedback: str):
    """Speak a feedback message at most once every SPEECH_COOLDOWN seconds."""
    now = time.time()
    last_time = last_spoken_times.get(feedback, 0)
    if now - last_time >= SPEECH_COOLDOWN:
        last_spoken_times[feedback] = now
        speech_queue.put(feedback)


# ===============================
# MediaPipe Setup
# ===============================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ===============================
# Angle Calculation
# ===============================

def calculate_angle(a, b, c) -> float:
    """Calculate the angle at point b given three 2D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


# ===============================
# State Variables
# ===============================

sequence = []
prediction_history = []
stable_prediction = "No exercise"

rep_count = 0
stage = None
curl_angle = 0.0
min_curl_angle = 180.0

start_elbow_x = None
elbow_drift = 0.0
elbow_drift_detected = False

error_counts = {
    "Keep Elbows Close To Torso": 0,
    "Complete Full Range": 0,
}


# ===============================
# Webcam Loop
# ===============================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    feedback_list = []
    movement = 0.0

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks
        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        sequence.append(landmarks)
        sequence = sequence[-30:]

        # ---------------------------
        # Motion Detection
        # ---------------------------
        if len(sequence) >= 2:
            movement = np.mean(np.abs(
                np.array(sequence[-1]) - np.array(sequence[-2])
            ))

            if movement < 0.01:
                cv2.putText(frame, "No exercise", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Movement: {movement:.4f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press ESC to Exit", (20, 420),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Exercise Recognition", frame)
                if cv2.waitKey(1) == 27:
                    break
                continue

        # ---------------------------
        # LSTM Prediction
        # ---------------------------
        if len(sequence) == 30:
            sample = np.expand_dims(np.array(sequence), axis=0)
            prediction = model.predict(sample, verbose=0)
            confidence = float(np.max(prediction))
            pred = int(np.argmax(prediction))

            if confidence < 0.5:
                feedback_list.append(
                        f"Wrong Exercise: {stable_prediction.replace('_', ' ').title()}"
                    )


            if confidence >= 0.85:
                exercise = encoder.inverse_transform([pred])[0]
                prediction_history.append(exercise)
                prediction_history = prediction_history[-10:]

                stable_prediction = (
                    max(set(prediction_history), key=prediction_history.count)
                    if len(prediction_history) == 10
                    else exercise
                )

                # ---------------------------
                # Bicep Curl Analysis
                # ---------------------------
                if stable_prediction != "bicep_curl":

                    feedback_list.append(
                        f"Wrong Exercise: {stable_prediction.replace('_', ' ').title()}"
                    )

    

                # ---------------------------
                # Bicep Curl Analysis
                # ---------------------------
                else:
                    lm = result.pose_landmarks.landmark

                    shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow    = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist    = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    curl_angle = calculate_angle(shoulder, elbow, wrist)
                    min_curl_angle = min(min_curl_angle, curl_angle)

                    # Elbow Drift Detection
                    if start_elbow_x is not None:
                        elbow_drift = abs(elbow[0] - start_elbow_x)
                        if elbow_drift > 0.05:
                            feedback_list.append("Wrong exercise")
                            elbow_drift_detected = True

                    # Rep Counting
                    if curl_angle > 150:

                        if stage != "DOWN":
                            start_elbow_x = elbow[0]

                        stage = "DOWN"

                    elif curl_angle < 50 and stage == "DOWN":
                        # Full Range of Motion Check
                        if min_curl_angle > 60:
                            feedback_list.append("Complete Full Range")
                            error_counts["Complete Full Range"] += 1

                        stage = "UP"
                        rep_count += 1
                        min_curl_angle = 180.0

                        if elbow_drift_detected:
                            error_counts["Keep Elbows Close To Torso"] += 1
                            elbow_drift_detected = False

                        print(f"Bicep Curl Rep Count: {rep_count}")

            # ---------------------------
            # Voice Feedback (once per 5s per message)
            # ---------------------------
            for fb in feedback_list:
                speak_feedback(fb)

            # ---------------------------
            # On-Screen Text
            # ---------------------------
            #cv2.putText(frame, f"{stable_prediction}: {confidence:.2f}", (20, 40),
                       # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Target: bicep_curl", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle: {min_curl_angle:.1f}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Reps: {rep_count}", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Stage: {stage}", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "ESC = Exit", (20, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #cv2.putText(frame, f"Movement: {movement:.4f}", (20, 80),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #cv2.putText(frame, f"Angle: {curl_angle:.1f}", (20, 120),
             #
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #cv2.putText(frame, f"Reps: {rep_count}", (20, 160),
             #           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            #cv2.putText(frame, f"Drift: {elbow_drift:.3f}", (20, 200),
             #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #cv2.putText(frame, f"Min Angle: {min_curl_angle:.1f}", (20, 240),
             #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for i, fb in enumerate(feedback_list):
                cv2.putText(frame, fb, (20, 290 + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    else:
        cv2.putText(frame, "Body not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "Press ESC to Exit", (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Exercise Recognition", frame)

    if cv2.waitKey(1) == 27:
        break


# ===============================
# Workout Summary
# ===============================

print("\n========== WORKOUT SUMMARY ==========")
print(f"Total Reps: {rep_count}")
for error, count in error_counts.items():
    print(f"  {error}: {count} time(s)")
print("=====================================\n")


# ===============================
# Cleanup
# ===============================

speech_queue.put(None)
speech_thread.join(timeout=2)
cap.release()
cv2.destroyAllWindows()