import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder

os.makedirs("processed",exist_ok=True)
os.makedirs("models",exist_ok=True)

# Load files
landmarks=pd.read_csv(
    "data/raw/landmarks.csv"
)

labels=pd.read_csv(
    "data/raw/labels.csv"
)

# Merge using vid_id
data=pd.merge(
    landmarks,
    labels,
    on='vid_id'
)

print(data.head())

# Convert class names to numbers
encoder=LabelEncoder()

data['class']=encoder.fit_transform(
    data['class']
)

pickle.dump(
    encoder,
    open(
        "models/encoder.pkl",
        "wb"
    )
)

print(
    encoder.classes_
)

# Remove columns not used as features
features=data.drop(
    ['vid_id','frame_order','class'],
    axis=1
)

sequence_length=30

X=[]
y=[]

# Group frames by video
groups=data.groupby('vid_id')

for vid,group in groups:

    group=group.sort_values(
        'frame_order'
    )

    feature_data=group.drop(
        ['vid_id','frame_order','class'],
        axis=1
    ).values

    label=group['class'].iloc[0]

    for i in range(
        len(feature_data)-sequence_length
    ):

        X.append(
            feature_data[
                i:i+sequence_length
            ]
        )

        y.append(label)

X=np.array(X)
y=np.array(y)

np.save(
    "processed/X.npy",
    X
)

np.save(
    "processed/y.npy",
    y
)

print("X shape:",X.shape)
print("y shape:",y.shape)