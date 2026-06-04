import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# ===============================
# Load model and encoder
# ===============================

model = load_model(
    "models/exercise_model.h5"
)

encoder = pickle.load(
    open(
        "models/encoder.pkl",
        "rb"
    )
)


# ===============================
# MediaPipe setup
# ===============================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


# ===============================
# Webcam
# ===============================

cap = cv2.VideoCapture(0)

sequence = []

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


            if confidence<0.85:

                exercise="No exercise"

            else:

                exercise=encoder.inverse_transform(
                    [pred]
                )[0]


            cv2.putText(
                frame,
                f"{exercise}: {confidence:.2f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
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
        (20,120),
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


cap.release()
cv2.destroyAllWindows()