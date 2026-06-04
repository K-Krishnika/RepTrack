import pandas as pd
import matplotlib.pyplot as plt

# Load your landmarks + labels
landmarks = pd.read_csv(
    "data/raw/landmarks.csv"
)

labels = pd.read_csv(
    "data/raw/labels.csv"
)

# Show available classes
print("\nAvailable classes:")
print(labels["class"].unique())

# Choose exercise
exercise = input(
    "\nEnter class name: "
)

# Find first sample of that exercise
sample_id = labels[
    labels["class"] == exercise
]["vid_id"].iloc[0]

print("Selected video id:",sample_id)

# Get first frame of that sample
frame = landmarks[
    landmarks["vid_id"]==sample_id
].iloc[0]

x=[]
y=[]

# Extract x,y coordinates
for i in range(2,101,3):

    x.append(
        frame.iloc[i]
    )

    y.append(
        frame.iloc[i+1]
    )


# Approximate MediaPipe connections
connections = [

(0,11),(0,12),
(11,13),(13,15),
(12,14),(14,16),

(11,23),(12,24),
(23,24),

(23,25),(25,27),
(27,31),

(24,26),(26,28),
(28,32)

]

plt.figure(figsize=(8,8))

# Draw joints
plt.scatter(x,y,s=100)

# Draw skeleton lines
for start,end in connections:

    if start<len(x) and end<len(x):

        plt.plot(
            [x[start],x[end]],
            [y[start],y[end]]
        )

plt.gca().invert_yaxis()

plt.title(
    f"Skeleton: {exercise}"
)

plt.show()