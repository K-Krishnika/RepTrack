import pickle
import matplotlib.pyplot as plt

history = pickle.load(
    open(
        "models/history.pkl",
        "rb"
    )
)

# =====================
# ACCURACY GRAPH
# =====================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Model Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(True)

plt.show()

# =====================
# LOSS GRAPH
# =====================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    history["loss"],
    label="Train Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Model Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(True)

plt.show()