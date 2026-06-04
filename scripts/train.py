import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout

X=np.load(
    "processed/X.npy"
)

y=np.load(
    "processed/y.npy"
)

X_train,X_test,y_train,y_test=\
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

num_classes=len(
    np.unique(y)
)

model=Sequential()

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
        activation='relu'
    )
)

model.add(
    Dense(
        num_classes,
        activation='softmax'
    )
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history=model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(
        X_test,
        y_test
    )
)

model.save(
    "models/exercise_model.h5"
)

loss,acc=model.evaluate(
    X_test,
    y_test
)

print(
    "Accuracy:",
    acc
)