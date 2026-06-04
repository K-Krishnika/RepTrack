import pandas as pd
import os

# =========================
# LOAD DATA
# =========================

old_landmarks = pd.read_csv(
    "data/raw/landmarks.csv"
)

old_labels = pd.read_csv(
    "data/raw/labels.csv"
)

new_landmarks = pd.read_csv(
    "data/bicep_features/landmarks_fixed.csv"
)

new_labels = pd.read_csv(
    "data/bicep_features/labels.csv"
)

# =========================
# VERIFY COLUMNS
# =========================

if list(old_landmarks.columns) != list(new_landmarks.columns):
    print("ERROR: Column order mismatch!")
    exit()

print("Columns verified ✓")

# =========================
# MERGE
# =========================

merged_landmarks = pd.concat(
    [old_landmarks, new_landmarks],
    ignore_index=True
)

merged_labels = pd.concat(
    [old_labels, new_labels],
    ignore_index=True
)

# =========================
# CREATE FOLDER
# =========================

os.makedirs(
    "data/merged",
    exist_ok=True
)

# =========================
# SAVE
# =========================

merged_landmarks.to_csv(
    "data/merged/landmarks.csv",
    index=False
)

merged_labels.to_csv(
    "data/merged/labels.csv",
    index=False
)

# =========================
# STATS
# =========================

print("\n===== MERGE COMPLETE =====")

print(
    "Merged landmarks:",
    merged_landmarks.shape
)

print(
    "Merged labels:",
    merged_labels.shape
)

print("\nClasses:\n")
print(
    merged_labels["class"].value_counts()
)

print("\nSaved:")
print("data/merged/landmarks.csv")
print("data/merged/labels.csv")