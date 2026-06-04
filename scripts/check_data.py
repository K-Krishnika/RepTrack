import pandas as pd

landmarks=pd.read_csv(
    "data/raw/landmarks.csv"
)

labels=pd.read_csv(
    "data/raw/labels.csv"
)

print(landmarks.head())

print()

print(labels.head())

print()

print(landmarks.shape)

print(labels.shape)


print("F")

df = pd.read_csv("data/raw/landmarks.csv")

cols = [
    "x_left_shoulder",
    "x_right_shoulder",
    "x_left_hip",
    "x_right_hip",
    "y_left_shoulder",
    "y_right_shoulder",
    "y_left_hip",
    "y_right_hip"
]

print(df[cols].head())

labels = pd.read_csv("data/raw/labels.csv")

print(labels["class"].value_counts())

df = pd.read_csv("data/raw/landmarks.csv")

counts = df.groupby("vid_id").size()

print(counts.describe())
print(counts.head())
