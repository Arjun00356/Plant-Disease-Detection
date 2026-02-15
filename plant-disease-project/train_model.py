import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "dataset/train"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# DATA GENERATOR
# ==============================
# Data augmentation improves generalization
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Training data
train_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical"
)

# Validation data
val_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

# ==============================
# SAVE CLASS NAMES (IMPORTANT)
# ==============================
class_indices = train_data.class_indices

# Convert dict index->class list
class_names = list(class_indices.keys())

with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)

print("✅ Class names saved.")

# ==============================
# MODEL BUILDING (TRANSFER LEARNING)
# ==============================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# CALLBACKS
# ==============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ==============================
# TRAINING
# ==============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==============================
# SAVE FINAL MODEL
# ==============================
model.save(os.path.join(MODEL_DIR, "plant_model.h5"))
print("✅ Model saved.")

# ==============================
# SAVE MODEL ACCURACY
# ==============================
final_val_acc = history.history["val_accuracy"][-1] * 100

with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump({"accuracy": round(final_val_acc, 2)}, f)

print(f"✅ Validation Accuracy: {final_val_acc:.2f}% saved.")

# ==============================
# OPTIONAL: TRAINING CURVES
# ==============================
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Accuracy")
plt.savefig(os.path.join(MODEL_DIR, "accuracy_graph.png"))
plt.show()

print("🎉 Training complete!")
