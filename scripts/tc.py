import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys
import time
import queue
import threading

from tensorflow.keras.models import load_model
import pyttsx3


# =====================================
# Text-to-Speech Setup
# =====================================

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

# Tracks last spoken time per feedback message
last_spoken_times = {}
SPEECH_COOLDOWN = 5.0  # seconds


def speak_feedback(feedback: str):
    """Speak feedback at most once every SPEECH_COOLDOWN seconds per unique message."""
    now = time.time()
    if now - last_spoken_times.get(feedback, 0) >= SPEECH_COOLDOWN:
        last_spoken_times[feedback] = now
        speech_queue.put(feedback)


# =====================================
# User Input
# =====================================

if len(sys.argv) == 3:
    target_exercise = sys.argv[1]
    target_reps = int(sys.argv[2])
else:
    target_exercise = input("Choose exercise (push_up / squat / bicep_curl): ").strip().lower()
    target_reps = int(input("Enter target reps: "))


# =====================================
# Load Model
# =====================================

model = load_model("models/exercise_model.h5")
encoder = pickle.load(open("models/encoder.pkl", "rb"))

print("\nAvailable classes:")
print(encoder.classes_)


# =====================================
# MediaPipe Setup
# =====================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =====================================
# Angle Calculation
# =====================================

def calculate_angle(a, b, c) -> float:
    """Calculate the angle at point b given three 2D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


# =====================================
# State Variables
# =====================================

rep_count = 0
stage = None
workout_complete = False

sequence = []
prediction_history = []

# Bicep curl state
start_elbow_x = None
elbow_drift = 0.0
elbow_drift_detected = False
min_curl_angle = 180.0

# Push-up state
min_pushup_angle = 180.0

error_counts = {
    "Keep Elbows Close To Torso": 0,
    "Complete Full Range": 0,
}

# Display variables (safe defaults)
stable_prediction = "No exercise"
confidence = 0.0
movement = 0.0
angle = 0.0
knee_distance = 0.0
ankle_distance = 0.0
torso_angle = 0.0
elbow_drift_display = 0.0
curl_angle = 0.0


# =====================================
# Webcam Loop
# =====================================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    feedback_list = []

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        sequence.append(landmarks)
        sequence = sequence[-30:]

        # ---------------------------
        # Movement Detection
        # ---------------------------
        if len(sequence) >= 2:
            movement = float(np.mean(np.abs(
                np.array(sequence[-1]) - np.array(sequence[-2])
            )))

        # ---------------------------
        # LSTM Prediction
        # ---------------------------
        if len(sequence) == 30:
            sample = np.expand_dims(np.array(sequence), axis=0)
            prediction = model.predict(sample, verbose=0)
            confidence = float(np.max(prediction))
            pred = int(np.argmax(prediction))

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
            # Rep Counter
            # ---------------------------
            # ---------------------------
# Check Target Exercise
# ---------------------------
            if stable_prediction != target_exercise:

                feedback_list.append(
                    f"Wrong Exercise"
                )

            elif stable_prediction == target_exercise:
                lm = result.pose_landmarks.landmark

                # -------------------------
                # Push-Up
                # -------------------------
                if target_exercise == "push_up":
                    shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow    = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist    = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    r_elbow    = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                  lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    r_wrist    = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                  lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    hip   = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    angle       = calculate_angle(shoulder, elbow, wrist)
                    right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    body_angle  = calculate_angle(shoulder, hip, ankle)

                    if stage == "DOWN":
                        min_pushup_angle = min(min_pushup_angle, angle)

                    if abs(angle - right_angle) > 20:
                        feedback_list.append("Push Evenly")
                    if body_angle < 160:
                        feedback_list.append("Keep Your Body Straight")
                    if hip[1] < shoulder[1] - 0.05:
                        feedback_list.append("Lower Your Hips")

                    if angle < 90:
                        stage = "DOWN"
                    elif angle > 160 and stage == "DOWN":
                        if min_pushup_angle > 100:
                            feedback_list.append("Lower Your Chest Further")
                        min_pushup_angle = 180.0
                        stage = "UP"
                        rep_count += 1
                        print(f"Push-Up Rep Count: {rep_count}")

                # -------------------------
                # Squat
                # -------------------------
                elif target_exercise == "squat":
                    hip      = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    knee     = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle    = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    l_knee  = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                               lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    r_knee  = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    angle        = calculate_angle(hip, knee, ankle)
                    torso_angle  = calculate_angle(shoulder, hip, knee)
                    knee_distance  = abs(l_knee[0] - r_knee[0])
                    ankle_distance = abs(l_ankle[0] - r_ankle[0])

                    if 120 <= angle < 140:
                        feedback_list.append("Go Lower")
                    if torso_angle < 150:
                        feedback_list.append("Straighten Your Back")
                    if angle < 120 and ankle_distance > 0 and knee_distance < ankle_distance * 0.7:
                        feedback_list.append("Keep Knees Aligned")

                    if angle > 150 and stage is None:
                        stage = "UP"
                    if angle < 120 and stage == "UP":
                        stage = "DOWN"
                    elif angle > 160 and stage == "DOWN":
                        stage = "UP"
                        rep_count += 1
                        print(f"Squat Rep Count: {rep_count}")

                # -------------------------
                # Bicep Curl
                # -------------------------
                elif target_exercise == "bicep_curl":
                    shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow    = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist    = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    curl_angle = calculate_angle(shoulder, elbow, wrist)
                    angle = curl_angle
                    min_curl_angle = min(min_curl_angle, curl_angle)
                    elbow_drift_display = 0.0

                    if start_elbow_x is not None:
                        elbow_drift_display = abs(elbow[0] - start_elbow_x)
                        if elbow_drift_display > 0.05:
                            feedback_list.append("Keep Elbows Close To Torso")
                            elbow_drift_detected = True

                    if curl_angle > 150:
                        stage = "DOWN"
                        if start_elbow_x is None:
                            start_elbow_x = elbow[0]
                    elif curl_angle < 50 and stage == "DOWN":
                        if min_curl_angle > 60:
                            feedback_list.append("Complete Full Range")
                            error_counts["Complete Full Range"] += 1
                        stage = "UP"
                        if elbow_drift_detected:
                            error_counts["Keep Elbows Close To Torso"] += 1
                            elbow_drift_detected = False
                        rep_count += 1
                        min_curl_angle = 180.0
                        print(f"Bicep Curl Rep Count: {rep_count}")

            # ---------------------------
            # Target Rep Check
            # ---------------------------
            if rep_count >= target_reps:
                workout_complete = True

        # ---------------------------
        # Voice Feedback (per message, 5s cooldown)
        # ---------------------------
        for fb in feedback_list:
            speak_feedback(fb)

    else:
        cv2.putText(frame, "Body Not Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # =====================================
    # Display
    # =====================================

    cv2.putText(frame, f"Detected: {stable_prediction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Target: {target_exercise}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Angle: {angle:.1f}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {rep_count}/{target_reps}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Stage: {stage}", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "ESC = Exit", (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Exercise-specific debug info
    if target_exercise == "bicep_curl":
        cv2.putText(frame, f"Drift: {elbow_drift_display:.3f}", (20, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Min Angle: {min_curl_angle:.1f}", (20, 540),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif target_exercise == "squat":
        cv2.putText(frame, f"Knee Dist: {knee_distance:.2f}", (20, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Ankle Dist: {ankle_distance:.2f}", (20, 540),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Feedback messages on screen
    for i, fb in enumerate(feedback_list):
        cv2.putText(frame, fb, (20, 320 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if workout_complete:
        cv2.putText(frame, "WORKOUT COMPLETE!", (50, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)

    cv2.imshow("Exercise Recognition", frame)

    if cv2.waitKey(1) == 27:
        break


# =====================================
# Cleanup
# =====================================

speech_queue.put(None)
speech_thread.join(timeout=2)
cap.release()
cv2.destroyAllWindows()