import numpy as np
import pickle

from tensorflow.keras.models import load_model

# Load model
model=load_model(
    "models/exercise_model.h5"
)

# Load encoder
encoder=pickle.load(
    open(
        "models/encoder.pkl",
        "rb"
    )
)

# Load processed data
X=np.load(
    "processed/X.npy"
)

y=np.load(
    "processed/y.npy"
)

# Take one sample
sample=X[100]

# LSTM expects batch dimension
sample=np.expand_dims(
    sample,
    axis=0
)

prediction=model.predict(
    sample
)

predicted_class=np.argmax(
    prediction
)

exercise=encoder.inverse_transform(
    [predicted_class]
)

print(
    "Prediction:",
    exercise[0]
)

print(
    "Confidence:",
    np.max(prediction)
)