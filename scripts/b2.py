import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import sys
import pyttsx3
import time


# =====================================
#Initialise text to speech engine
# =====================================

engine = pyttsx3.init()

engine.setProperty(
    "rate",
    150
)

last_spoken_feedback = ""
last_speech_time = 0


# =====================================
# USER INPUT
# =====================================


if len(sys.argv) == 3:

    target_exercise = sys.argv[1]
    target_reps = int(sys.argv[2])

else:

    target_exercise = input(
        "Choose exercise: "
    ).strip().lower()

    target_reps = int(
        input(
            "Enter target reps: "
        )
    )
# =====================================
# LOAD MODEL
# =====================================

model = load_model(
    "models/exercise_model.h5"
)

encoder = pickle.load(
    open(
        "models/encoder.pkl",
        "rb"
    )
)

print("\nAvailable classes:")
print(encoder.classes_)

# =====================================
# MEDIAPIPE
# =====================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# =====================================
# VARIABLES
# =====================================

rep_count = 0
stage = None
workout_complete = False

sequence = []
prediction_history = []

# =========================
# BICEP VARIABLES
# =========================

start_elbow_x = None
elbow_drift = 0
elbow_drift_detected = False

min_curl_angle = 180

error_counts = {
    "Keep Elbows Close To Torso": 0,
    "Complete Full Range": 0
}

# =====================================
# ANGLE FUNCTION
# =====================================

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

# =====================================
# WEBCAM
# =====================================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = pose.process(rgb)

    stable_prediction = "No exercise"
    confidence = 0.0
    movement = 0.0
    angle = 0.0
    body_angle = 0.0
    knee_distance = 0
    ankle_distance = 0
    torso_angle = 0
    min_pushup_angle = 180
    feedback = ""
    feedback_list = []

    if result.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = []

        for lm in result.pose_landmarks.landmark:

            landmarks.extend(
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
            )

        sequence.append(
            landmarks
        )

        sequence = sequence[-30:]

        # =====================================
        # MOVEMENT DETECTION
        # =====================================

        if len(sequence) >= 2:

            movement = np.mean(
                np.abs(
                    np.array(sequence[-1])
                    -
                    np.array(sequence[-2])
                )
            )

        # =====================================
        # PREDICTION
        # =====================================

        if len(sequence) == 30:

            sample = np.expand_dims(
                np.array(sequence),
                axis=0
            )

            prediction = model.predict(
                sample,
                verbose=0
            )

            confidence = np.max(
                prediction
            )

            pred = np.argmax(
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

            # =====================================
            # REP COUNTER
            # =====================================

            if stable_prediction == target_exercise:

                lm = result.pose_landmarks.landmark

                # =================================
                # PUSH-UP COUNTER
                # =================================

                if target_exercise == "push_up":

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

                    right_shoulder = [
                        lm[
                            mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                        ].y
                    ]

                    right_elbow = [
                        lm[
                            mp_pose.PoseLandmark.RIGHT_ELBOW.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.RIGHT_ELBOW.value
                        ].y
                    ]

                    right_wrist = [
                        lm[
                            mp_pose.PoseLandmark.RIGHT_WRIST.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.RIGHT_WRIST.value
                        ].y
                    ]

                    hip = [
                        lm[
                            mp_pose.PoseLandmark.LEFT_HIP.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.LEFT_HIP.value
                        ].y
                    ]

                    ankle = [
                        lm[
                            mp_pose.PoseLandmark.LEFT_ANKLE.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.LEFT_ANKLE.value
                        ].y
                    ]

                    body_angle = calculate_angle(
                        shoulder,
                        hip,
                        ankle
                    )

                    right_angle = calculate_angle(
                        right_shoulder,
                        right_elbow,
                        right_wrist
                    )

                    angle = calculate_angle(
                        shoulder,
                        elbow,
                        wrist
                    )

                    # =========================
                    # PUSHUP ROM CHECK
                    # =========================

                    if stage == "DOWN":
                        min_pushup_angle = min(
                            min_pushup_angle,
                            angle
                        )

                    if abs(angle - right_angle) > 20:
                        feedback_list.append(
                            "Push Evenly"
                        )

                    if body_angle < 160:

                        feedback_list.append(
                            "Keep Your Body Straight"
                        )

                    if hip[1] < shoulder[1] - 0.05:
                        feedback_list.append(
                            "Lower Your Hips"
                        )

                    if angle < 90:
                        stage = "DOWN"

                    elif angle > 160 and stage == "DOWN":

                        if min_pushup_angle > 100:
                            feedback_list.append(
                                "Lower Your Chest Further"
                            )

                        min_pushup_angle = 180

                        stage = "UP"
                        rep_count += 1

                # =================================
                # SQUAT COUNTER
                # =================================

                elif target_exercise == "squat":

                    hip = [
                        lm[
                            mp_pose.PoseLandmark.LEFT_HIP.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.LEFT_HIP.value
                        ].y
                    ]

                    shoulder = [
                        lm[
                            mp_pose.PoseLandmark.LEFT_SHOULDER.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.LEFT_SHOULDER.value
                        ].y
                    ]

                    knee = [
                        lm[
                            mp_pose.PoseLandmark.LEFT_KNEE.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.LEFT_KNEE.value
                        ].y
                    ]

                    left_knee = [
                        lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                    ]       
                    
                    right_knee = [
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
                    ]

                    left_ankle = [
                        lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                    ]

                    right_ankle = [
                        lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                    ]

                    ankle = [
                        lm[
                            mp_pose.PoseLandmark.LEFT_ANKLE.value
                        ].x,
                        lm[
                            mp_pose.PoseLandmark.LEFT_ANKLE.value
                        ].y
                    ]

                    angle = calculate_angle(
                        hip,
                        knee,
                        ankle
                    )

                    torso_angle = calculate_angle(
                        shoulder,
                        hip,
                        knee
                    )

                    # =========================
                    # KNEE VALGUS
                    # =========================

                    knee_distance = abs(
                        left_knee[0] - right_knee[0]
                    )

                    ankle_distance = abs(
                        left_ankle[0] - right_ankle[0]
                    )

                    #Squat depth

                    if angle >= 120 and angle < 140:
                        feedback_list.append("Go Lower")

                    # =========================
                    # FORWARD LEAN
                    # =========================

                    if torso_angle < 150:
                        feedback_list.append("Straighten Your Back")

                    # =========================
                    # KNEE VALGUS
                    # Only check near squat bottom
                    # =========================

                    if angle < 120:

                        if (
                            ankle_distance > 0 and
                            knee_distance < ankle_distance * 0.7
                        ):
                            feedback_list.append(
                                "Keep Knees Aligned"
                            )

                    # Standing position

                    if angle > 150 and stage is None:
                        stage = "UP"

                    # Going down

                    if angle < 120 and stage == "UP":
                        stage = "DOWN"

                    # Coming back up = count rep

                    elif angle > 160 and stage == "DOWN":

                        stage = "UP"

                        rep_count += 1

                        print(
                            f"Squat Rep Count: {rep_count}"
                        )

                elif target_exercise == "bicep_curl":
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

            # =====================================
            # TARGET COMPLETE
            # =====================================

            if rep_count >= target_reps:

                workout_complete = True

    else:

        cv2.putText(
            frame,
            "Body Not Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # =====================================
    # DISPLAY
    # =====================================

    cv2.putText(
        frame,
        f"Detected: {stable_prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Target: {target_exercise}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Angle: {angle:.1f}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if target_exercise == "bicep_curl":

        cv2.putText(
            frame,
            f"Drift:{elbow_drift:.3f}",
            (20, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"MinAngle:{min_curl_angle:.1f}",
            (20, 540),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

    cv2.putText(
        frame,
        f"Reps: {rep_count}/{target_reps}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Stage: {stage}",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"KneeDist:{knee_distance:.2f}",
        (20, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"AnkleDist:{ankle_distance:.2f}",
        (20, 540),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    current_time = time.time()


    if (
        len(feedback_list) > 0
        and
        current_time - last_speech_time > 5
    ):
        

        engine.say(
            feedback_list[0]
        )

        engine.runAndWait()

        last_speech_time = current_time

    for i, feedback in enumerate(feedback_list):

        # =========================
        # VOICE FEEDBACK
        # =========================


        color = (0, 255, 0)

        if feedback in [
            "Go Lower",
            "Straighten Your Back",
            "Keep Knees Aligned",
            "Keep Elbows Close To Torso",
            "Complete Full Range",
            "Lower Your Chest Further",
            "Keep Your Body Straight",
            "Push Evenly",
            "Lower Your Hips"
        ]:
            color = (0, 0, 255)

        cv2.putText(
            frame,
            feedback,
            (20, 320 + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

    if workout_complete:

        cv2.putText(
            frame,
            "WORKOUT COMPLETE!",
            (50, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            4
        )

    cv2.putText(
        frame,
        "ESC = Exit",
        (20, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    

    cv2.imshow(
        "Exercise Recognition",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()