import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# LOAD
# =========================

landmarks = pd.read_csv(
    "data/merged/landmarks.csv"
)

labels = pd.read_csv(
    "data/merged/labels.csv"
)

# =========================
# MERGE
# =========================

data = landmarks.merge(
    labels,
    on="vid_id"
)

# =========================
# KEEP ONLY 3 EXERCISES
# =========================

target_classes = [
    "push_up",
    "squat",
    "bicep_curl"
]

data = data[
    data["class"].isin(target_classes)
]

print(data["class"].value_counts())

# =========================
# FEATURES
# =========================

X = data.drop(
    columns=[
        "vid_id",
        "frame_order",
        "class"
    ]
)

y = data["class"]

# =========================
# ENCODE LABELS
# =========================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print("Classes:")
print(encoder.classes_)

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("Training...")

model.fit(
    X_train,
    y_train
)

# =========================
# EVALUATION
# =========================

pred = model.predict(X_test)

print(
    classification_report(
        y_test,
        pred,
        target_names=encoder.classes_
    )
)

# =========================
# SAVE
# =========================

joblib.dump(
    model,
    "models/exercise_classifier.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("\nSaved!")
print("models/exercise_classifier.pkl")
print("models/label_encoder.pkl")