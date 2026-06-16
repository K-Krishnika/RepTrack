import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

# ==========================
# LOAD DATA
# ==========================

X = np.load(
    "processed/X.npy"
)

y = np.load(
    "processed/y.npy"
)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# MODEL
# ==========================

num_classes = len(
    np.unique(y)
)

model = Sequential()

model.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(
            X.shape[1],
            X.shape[2]
        )
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(64)
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        num_classes,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# TRAIN
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(
        X_test,
        y_test
    )
)

# ==========================
# SAVE MODEL
# ==========================

model.save(
    "models/exercise_model.h5"
)

# ==========================
# SAVE HISTORY
# ==========================

with open(
    "models/history.pkl",
    "wb"
) as f:

    pickle.dump(
        history.history,
        f
    )

# ==========================
# EVALUATION
# ==========================

y_prob = model.predict(
    X_test
)

y_pred = np.argmax(
    y_prob,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n========== RESULTS ==========")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)

np.save(
    "models/confusion_matrix.npy",
    cm
)

print(
    "\nConfusion Matrix Saved."
)