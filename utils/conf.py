import numpy as np
import pickle

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Load Files
# ==========================

model = load_model("models\exercise_model.h5")
encoder = pickle.load(open("models\encoder.pkl", "rb"))

X = np.load("processed\X.npy")
Y = np.load("processed\Y.npy")

# ==========================
# Predictions
# ==========================

y_pred = np.argmax(model.predict(X), axis=1)

if len(Y.shape) > 1:
    y_true = np.argmax(Y, axis=1)
else:
    y_true = Y

# ==========================
# Class Indices
# ==========================

pushup_idx = list(encoder.classes_).index('push_up')
squat_idx = list(encoder.classes_).index('squat')

selected_labels = [pushup_idx, squat_idx]

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=selected_labels
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Push-Up', 'Squat'],
    yticklabels=['Push-Up', 'Squat']
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("pushup_squat_confusion_matrix.png", dpi=300)
plt.show()