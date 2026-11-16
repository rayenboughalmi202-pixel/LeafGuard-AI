# train.py
# This file loads the leaf images, builds a small CNN,
# trains it for a few epochs and saves it to a file.

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# All images are resized to this size before training.
IMAGE_SIZE = (128, 128)

# Folder that holds the two sub-folders: healthy/ and diseased/
DATASET_FOLDER = "dataset"

# The file where the trained model is saved.
MODEL_FILE = "leafguard_model.keras"

# We have 2 classes: 0 = Healthy, 1 = Diseased
CLASS_NAMES = ["Healthy", "Diseased"]


def load_images(folder, label):
    """Load every image in one folder and give them all the same label."""
    images = []
    labels = []
    # os.listdir gives us every file name inside the folder.
    for file_name in os.listdir(folder):
        # Skip hidden files (like .gitkeep) and non-images.
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, file_name)
        # Open the image, force RGB colors, resize to 128x128.
        image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
        # Turn the image into numbers from 0 to 1 (instead of 0 to 255).
        # Neural networks learn better with small numbers.
        array = np.array(image) / 255.0
        images.append(array)
        labels.append(label)
    return images, labels


def load_dataset():
    """Load all images from dataset/healthy and dataset/diseased."""
    images, labels = [], []
    # Healthy leaves get label 0.
    h_images, h_labels = load_images(os.path.join(DATASET_FOLDER, "healthy"), 0)
    images += h_images
    labels += h_labels
    # Diseased leaves get label 1.
    d_images, d_labels = load_images(os.path.join(DATASET_FOLDER, "diseased"), 1)
    images += d_images
    labels += d_labels
    return np.array(images), np.array(labels)


def build_model():
    """Build a very small CNN: one convolution block + two dense layers."""
    model = models.Sequential()
    # Conv2D: looks for small patterns (edges, spots, colors).
    # 32 filters, each one looks at a 3x3 area of the image.
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
    # MaxPooling2D: shrinks the image by 2, keeps only the strongest signals.
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten: turns the 2D picture into a simple list of numbers.
    model.add(layers.Flatten())
    # Dense: a normal fully-connected layer with 64 neurons.
    model.add(layers.Dense(64, activation="relu"))
    # Last layer: 2 outputs (healthy / diseased) with softmax,
    # so the two numbers add up to 1 (like probabilities).
    model.add(layers.Dense(2, activation="softmax"))
    # adam = the optimizer (how the model improves).
    # The loss tells the model how wrong its predictions are.
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train():
    """Load the data, train the model for 5 epochs, save it."""
    print("Loading dataset...")
    images, labels = load_dataset()

    # We need at least 1 image in each class, or there is nothing to learn.
    if len(images) == 0:
        print("No images found in the dataset folder.")
        print("Add .jpg or .png leaf photos to dataset/healthy/ and dataset/diseased/")
        print("and run the training again.")
        return
    if len(set(labels.tolist())) < 2:
        print("Only one class has images. Add images to BOTH folders:")
        print("dataset/healthy/ and dataset/diseased/")
        return

    print(f"Found {len(images)} images.")

    # Shuffle the images so healthy and diseased are mixed.
    order = np.random.permutation(len(images))
    images, labels = images[order], labels[order]

    # Use 80% of the images for training, 20% for testing.
    # With a very small dataset the test set may be tiny or empty.
    split = int(len(images) * 0.8)
    x_train, y_train = images[:split], labels[:split]
    x_test, y_test = images[split:], labels[split:]

    model = build_model()
    print("Training... (5 epochs)")
    if len(x_test) > 0:
        # validation_data lets us see the accuracy on images
        # the model did NOT train on.
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    else:
        model.fit(x_train, y_train, epochs=5)

    # Final accuracy on the test images (if we have any).
    if len(x_test) > 0:
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Final test accuracy: {accuracy * 100:.1f}%")
    else:
        print("Dataset too small for a test split, "
              "accuracy shown above is training accuracy.")

    model.save(MODEL_FILE)
    print(f"Model saved as {MODEL_FILE}")


# This runs only when we start this file directly: python train.py
if __name__ == "__main__":
    train()
