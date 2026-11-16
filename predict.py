# predict.py
# This file loads the trained model and predicts one leaf image.

import os
import numpy as np
from PIL import Image
import tensorflow as tf

# The trained model file (created by train.py).
MODEL_FILE = "leafguard_model.keras"

# Must be in the same order as in train.py: 0 = Healthy, 1 = Diseased
CLASS_NAMES = ["Healthy", "Diseased"]


def predict(image_path):
    """Predict if one leaf image is healthy or diseased."""
    # 1. The model must exist. If not, train first.
    if not os.path.exists(MODEL_FILE):
        print(f"No trained model found ({MODEL_FILE}).")
        print("Run option 1 (Train the model) first.")
        return

    # 2. The image must exist.
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_FILE)

    # 3. Open the image, resize it exactly like the training images.
    image = Image.open(image_path).convert("RGB").resize((128, 128))
    array = np.array(image) / 255.0

    # 4. The model expects a "batch" of images, so we add one extra dimension.
    #    Shape goes from (128, 128, 3) to (1, 128, 128, 3).
    array = np.expand_dims(array, axis=0)

    # 5. Ask the model. It returns 2 numbers (probabilities) that add up to 1.
    result = model.predict(array, verbose=0)[0]
    best = int(np.argmax(result))             # index of the highest number
    confidence = float(result[best]) * 100    # turn it into a percentage

    print("=================================")
    print("Prediction")
    print("=================================")
    print(f"Image: {image_path}")
    print(f"Prediction: {CLASS_NAMES[best]}")
    print(f"Confidence: {confidence:.1f}%")


# This runs only when we start this file directly: python predict.py
if __name__ == "__main__":
    path = input("Enter the path of the leaf image: ").strip()
    predict(path)
