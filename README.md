# LeafGuard AI 🌿

A **very simple beginner-friendly AI project**: a plant leaf disease classifier
that runs entirely in the terminal.

Give it a photo of a plant leaf, and it predicts **Healthy** or **Diseased**
with a confidence percentage.

No website, no database, no API — just Python, a small neural network,
and your terminal.

---

## 1. What is LeafGuard AI?

LeafGuard AI is a small image classifier built with **TensorFlow/Keras**.

- The user gives the path of a leaf image.
- The program resizes the image to 128x128.
- A small CNN (convolutional neural network) predicts if the leaf
  is healthy or diseased.

It is a **student demonstration project**, made to be read, understood
and explained — not a professional farming system.

---

## 2. How to install

You need **Python 3.10 or newer**.

```bash
# 1. Go into the project folder
cd LeafGuard-AI

# 2. (Recommended) Create a virtual environment
python -m venv venv

# 3. Activate it
# On Windows:
venv\Scripts\activate
# On Mac / Linux:
source venv/bin/activate

# 4. Install the 3 libraries
pip install -r requirements.txt
```

That installs only: `tensorflow`, `pillow`, `numpy`.

---

## 3. How to organize the dataset

Put your leaf photos in the `dataset` folder, sorted by class:

```
dataset/
    healthy/
        leaf1.jpg
        leaf2.jpg
        ...
    diseased/
        leaf1.jpg
        leaf2.jpg
        ...
```

- **healthy/** — photos of healthy leaves.
- **diseased/** — photos of leaves with visible disease (spots, yellowing, mold...).
- Use `.jpg` or `.png` images. Any size works, the program resizes them to 128x128.
- Try to have **at least 20 images in each folder**. More images = better accuracy.
- The two folders should have roughly the same number of images.

### How to add more images

1. Take or download clear photos of leaves (one leaf per photo, good light).
2. Save healthy leaves in `dataset/healthy/` and sick leaves in `dataset/diseased/`.
3. Run option 1 (Train the model) again — the model trains from scratch
   on all the images, old and new.

### Where to find free leaf images

- The **PlantVillage** dataset (Kaggle / TensorFlow Datasets) has thousands
  of labeled leaf photos, free for learning projects.
- You can also take photos yourself with a phone.

---

## 4. How to train the model

```bash
python main.py
```

Choose **1. Train the model**.

The program will:

1. Load all images from `dataset/healthy/` and `dataset/diseased/`
2. Resize them to 128x128 and scale pixel values to 0–1
3. Train a small CNN for **5 epochs**
4. Show the accuracy
5. Save the trained model as `leafguard_model.keras`

Training takes a few seconds with a small dataset, a few minutes with
hundreds of images.

---

## 5. How to make a prediction

```bash
python main.py
```

Choose **2. Predict a leaf disease**, then type the path of an image:

```
Enter the path of the leaf image: dataset/diseased/leaf1.jpg
```

The program loads `leafguard_model.keras`, prepares the image
and prints the result.

> You must train the model (option 1) at least once before predicting.
> If no model file exists, the program will tell you to train first.

---

## 6. Example terminal output

```
=================================
        LeafGuard AI 🌿
=================================

1. Train the model
2. Predict a leaf disease
3. Exit

Choose an option: 1
Loading dataset...
Found 40 images.
Training... (5 epochs)
Epoch 1/5 ...
Epoch 5/5 ...
Final test accuracy: 87.5%
Model saved as leafguard_model.keras

=================================
        LeafGuard AI 🌿
=================================

1. Train the model
2. Predict a leaf disease
3. Exit

Choose an option: 2
Enter the path of the leaf image: dataset/diseased/leaf1.jpg
Loading model...
=================================
Prediction
=================================
Image: dataset/diseased/leaf1.jpg
Prediction: Diseased
Confidence: 87.4%
```

---

## 7. Limitations (read this before showing the project)

- **Only 2 classes**: Healthy vs Diseased. It cannot name the exact disease
  (for example "early blight" or "rust"). That would need more classes
  and much more data.
- **Accuracy depends on the dataset**: with 10 images the model mostly
  guesses; with hundreds of varied images it gets much better.
- **Simple model**: one convolution layer only. It learns colors, spots
  and shapes, not expert plant knowledge.
- **Same kind of photos**: the model works best on photos similar to the
  training photos (same lighting, one leaf, close-up). A photo of a whole
  field will confuse it.
- **No medical/agricultural guarantee**: this is a learning demo,
  not a tool for real farming decisions.

---

## Project structure

```
LeafGuard-AI/
│
├── dataset/
│   ├── healthy/        # healthy leaf photos (.jpg / .png)
│   └── diseased/       # diseased leaf photos (.jpg / .png)
│
├── train.py            # loads images, builds + trains the CNN, saves it
├── predict.py          # loads the saved model, predicts one image
├── main.py             # terminal menu (train / predict / exit)
├── requirements.txt    # the 3 libraries to install
├── leafguard_model.keras  # created after training (not in the repo)
└── README.md
```

## How the AI works (simple version)

1. **Image → numbers**: every photo is resized to 128x128 pixels and
   each pixel (red, green, blue) becomes a number between 0 and 1.
2. **Conv2D layer**: 32 small 3x3 filters slide over the image and learn
   to detect patterns — edges, spots, color changes.
3. **MaxPooling**: shrinks the image by 2, keeping only the strongest
   signals, so the model focuses on what matters.
4. **Flatten + Dense**: the patterns become a list of numbers, then a
   layer of 64 neurons combines them.
5. **Softmax output**: the last layer gives 2 probabilities (healthy /
   diseased) that add up to 100%. The higher one wins.
6. **Training**: the model sees the images 5 times (5 epochs) and
   slowly adjusts its internal numbers to make fewer mistakes.
