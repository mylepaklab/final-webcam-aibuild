import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.preprocessing import LabelEncoder

# --- Parameters ---
dataset_dir = "./custom_dataset_img"  # Path to your dataset folder
dataset_dir_train = "./custom_dataset_img/train"  # Path to your dataset folder
dataset_dir_val = "./custom_dataset_img/val"  # Path to your dataset folder
img_height = 256
img_width = 256
batch_size = 16

# --- 1. Load dataset from folders, automatically labeled ---
# This function splits into train and test sets (80/20)
train_ds = image_dataset_from_directory(
    dataset_dir_train,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'  # integer encoded labels
)

val_ds = image_dataset_from_directory(
    dataset_dir_val,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

# Get class names and save label mapping
class_names = train_ds.class_names
label_map = {name: idx for idx, name in enumerate(class_names)}
with open("label_mapping.json", "w") as f:
    json.dump(label_map, f, indent=2)
print(f"✅ Label mapping saved: {label_map}")

# --- 2. Build CNN Model ---
num_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Train Model ---
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# --- 4. Save Model Architecture and Weights ---
model_json = model.to_json()
with open("gesture_cnn_model.json", "w") as json_file:
    json_file.write(model_json)
print("✅ Model architecture saved to gesture_cnn_model.json")

# --- 4. Save Full Model for TFJS ---
model.save("gesture_model.h5")  # ✅ This saves architecture + weights together
print("✅ Full model saved to gesture_model.h5 (for TensorFlow.js conversion)")

# Save weights with the correct filename extension
model.save_weights("gesture_cnn_weights.weights.h5")
print("✅ Model weights saved to gesture_cnn_weights.weights.h5")

# --- 5. Example: Loading the model later ---

# To load architecture
# with open("gesture_cnn_model.json", "r") as json_file:
#     loaded_json = json_file.read()
# loaded_model = model_from_json(loaded_json)

# Load weights
# loaded_model.load_weights("gesture_cnn_weights.bin")

# Compile before use
# loaded_model.compile(optimizer='adam',
#                      loss='sparse_categorical_crossentropy',
#                      metrics=['accuracy'])
