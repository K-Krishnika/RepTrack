import numpy as np
import pickle

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# ==========================
# Load Model and Data
# ==========================

model = load_model("models\exercise_model.h5")
encoder = pickle.load(open("models\encoder.pkl", "rb"))

X = np.load("processed\X.npy")
Y = np.load("processed\Y.npy")

# ==========================
# Predictions
# ==========================

y_pred_prob = model.predict(X)
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert one-hot labels if necessary
if len(Y.shape) > 1:
    y_true = np.argmax(Y, axis=1)
else:
    y_true = Y

# ==========================
# Check Available Classes
# ==========================

print("\nAvailable Classes:")
print(encoder.classes_)

# ==========================
# Get Push-Up and Squat Indices
# ==========================

pushup_idx = list(encoder.classes_).index('push_up')
squat_idx = list(encoder.classes_).index('squat')

# ==========================
# Classification Report
# ==========================

report = classification_report(
    y_true,
    y_pred,
    labels=[pushup_idx, squat_idx],
    target_names=['Push-Up', 'Squat']
)

print("\n===== PUSH-UP & SQUAT REPORT =====\n")
print(report)