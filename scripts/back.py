import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# =====================================
# USER INPUT
# =====================================

target_exercise = input(
    "Enter exercise (push_up): "
).strip().lower()

target_reps = int(
    input("Enter target reps: ")
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
# REP COUNTER VARIABLES
# =====================================

rep_count = 0
stage = None
workout_complete = False

# =====================================
# SEQUENCE STORAGE
# =====================================

sequence = []
prediction_history = []

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
            # PUSHUP REP COUNTER
            # =====================================

            if (
                stable_prediction ==
                target_exercise
            ):

                lm = (
                    result.pose_landmarks.landmark
                )

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

                angle = calculate_angle(
                    shoulder,
                    elbow,
                    wrist
                )

                # DOWN

                if angle < 90:
                    stage = "DOWN"

                # UP

                if (
                    angle > 160
                    and stage == "DOWN"
                ):

                    stage = "UP"

                    rep_count += 1

                    print(
                        f"Rep Count: {rep_count}"
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
        f"Exercise: {stable_prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Movement: {movement:.4f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Reps: {rep_count}/{target_reps}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Stage: {stage}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    if workout_complete:

        cv2.putText(
            frame,
            "WORKOUT COMPLETE!",
            (80, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4
        )

    cv2.putText(
        frame,
        "Press ESC to Exit",
        (20, 240),
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