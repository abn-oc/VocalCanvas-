import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from audio_processor import UnifiedAudioProcessor

# =========================================================
# Config
# =========================================================
PROC = UnifiedAudioProcessor()

SPECTROGRAMS_DIR = "spectrograms"

SPEAKERS = [
    "abi",
    "ahmed",
    "zoha"
]

CNN_MODEL_PATH  = "models/vocalcanvas_cnn.keras"

os.makedirs("models", exist_ok=True)

IMG_HEIGHT = PROC.IMG_SIZE
IMG_WIDTH  = PROC.IMG_SIZE

# =========================================================
# CNN data loader — loads pre-saved PNG spectrograms
# =========================================================
def load_split(split_name):
    """
    Returns
    -------
    X_img : np.ndarray  (N, 128, 128, 1) float32 [0,1]
    y     : np.ndarray  (N,)             int
    """
    X_img = []
    y     = []

    for label, speaker in enumerate(SPEAKERS):

        folder = os.path.join(
            SPECTROGRAMS_DIR,
            split_name,
            speaker
        )

        files = [
            f for f in os.listdir(folder)
            if f.endswith(".png")
        ]

        for file in files:

            path = os.path.join(folder, file)

            img = load_img(
                path,
                color_mode="grayscale",
                target_size=(IMG_HEIGHT, IMG_WIDTH)
            )

            img_array = img_to_array(img) / 255.0   # (128, 128, 1)

            X_img.append(img_array)
            y.append(label)

    return np.array(X_img), np.array(y)





# =========================================================
# SECTION 1 — CNN  (Supervised)
# =========================================================
print("\n" + "=" * 60)
print("Training Convolutional Neural Network (CNN)")
print("=" * 60)

print("\nLoading spectrogram images...")

# Load our training, validation, and testing images
X_train_img, y_train = load_split("train")
X_val_img,   y_val   = load_split("val")
X_test_img,  y_test  = load_split("test")

print(
    f"Train size: {len(X_train_img)} | "
    f"Val size: {len(X_val_img)} | "
    f"Test size: {len(X_test_img)}"
)

# --- Class weights ---
# Helps the model focus on speakers with less data so no one gets left behind!
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights applied to handle any imbalanced data: {class_weight_dict}")

# --- Build CNN ---
print("\nBuilding CNN Model...")

cnn_model = models.Sequential([
    # Tell the model what shape the input images are
    Input(shape=X_train_img.shape[1:]),

    # 1st Convolutional block: Looks for simple patterns like edges or flat regions
    layers.Conv2D(32,  (3, 3), activation="relu"),
    layers.BatchNormalization(), # Keeps the numbers stable
    layers.MaxPooling2D((2, 2)), # Shrinks the image to focus on the most important parts
    layers.Dropout(0.3), # Randomly turns off some neurons to prevent memorizing the data (overfitting)

    # 2nd block: Looks for more complex patterns like shapes
    layers.Conv2D(64,  (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # 3rd block: Looks for highly specific features unique to each speaker
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Flatten the 2D images into a 1D list of numbers
    layers.Flatten(),
    
    # Hidden layer to process everything learned so far
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),

    # Output layer: One node for each speaker, showing the probability they are the speaker
    layers.Dense(len(SPEAKERS), activation="softmax"),
])

cnn_model.summary()

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy", # Good for picking exactly one winner out of many
    metrics=["accuracy"]
)

# --- Train ---
print("\nTraining CNN...")

# Start training the model!
history = cnn_model.fit(
    X_train_img, y_train,
    validation_data=(X_val_img, y_val),
    epochs=50, # Maximum number of passes over the dataset
    batch_size=16, # Process 16 images at a time
    class_weight=class_weight_dict,
    callbacks=[
        # Stop early if the model isn't getting better for 8 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True # Always keep the best version of the model
        ),
        # Slow down learning if we get stuck to fine-tune our approach
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
)

# --- Evaluate ---
print("\nEvaluating CNN on test set...")

_, test_acc = cnn_model.evaluate(X_test_img, y_test, verbose=0)
cnn_y_pred  = np.argmax(cnn_model.predict(X_test_img, verbose=0), axis=1)

print(f"CNN Test Accuracy: {test_acc * 100:.2f}%")
print("\nCNN Classification Report:")
print(
    classification_report(
        y_test, cnn_y_pred,
        target_names=SPEAKERS,
        zero_division=0
    )
)

# --- Confusion Matrix ---
cm_cnn = confusion_matrix(y_test, cnn_y_pred)
disp   = ConfusionMatrixDisplay(cm_cnn, display_labels=SPEAKERS)
disp.plot(cmap="Blues")
plt.title("VocalCanvas — CNN Confusion Matrix")
plt.savefig("confusion_matrix_cnn.png", dpi=150)
plt.show()

# --- Training curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"],     label="Train")
ax1.plot(history.history["val_accuracy"], label="Validation")
ax1.set_title("Accuracy"); ax1.legend()
ax2.plot(history.history["loss"],         label="Train")
ax2.plot(history.history["val_loss"],     label="Validation")
ax2.set_title("Loss"); ax2.legend()
plt.suptitle("VocalCanvas — CNN Training Curves")
plt.savefig("training_curves.png", dpi=150)
plt.show()

# --- Save ---
cnn_model.save(CNN_MODEL_PATH)
print(f"\nCNN saved → {CNN_MODEL_PATH}")

print("\n✅  Training complete.")
print(f"   CNN model   → {CNN_MODEL_PATH}")