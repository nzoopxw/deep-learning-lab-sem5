# Lab 03 — CNN for CIFAR-10 Image Classification

Design and implementation of a Convolutional Neural Network for 10-class image
classification on **CIFAR-10**, built with **TensorFlow/Keras**. This is
Experiment 3 of the CS3807 Deep Learning Laboratory (Semester V, AI & DS).

The lab walks through the full pipeline: exploring the dataset, building a
stacked conv–ReLU–pool CNN, training it, visualizing what the filters learn,
evaluating with standard metrics, and running a controlled max-vs-average
pooling comparison.

---

## Highlights

- **402,986** trainable parameters across 5 conv layers + a dense head
- **Test accuracy: 0.8361** (macro F1 0.8362) after 100 epochs with Adam
- Feature-map visualization from the first conv layer
- Full evaluation: precision / recall / F1, confusion matrix, classification report
- Max pooling vs. average pooling comparison (avg pooling edged ahead here: val acc 0.8431 vs 0.8361)

---

## Model Architecture

A `Sequential` model of three conv blocks (each `Conv2D → Conv2D → MaxPooling2D → Dropout`)
with a progressively widening filter count, followed by a dense head:

| Stage | Layers | Output shape |
|-------|--------|--------------|
| Block 1 | Conv2D(32) → Conv2D(32) → MaxPool → Dropout | 16×16×32 |
| Block 2 | Conv2D(64) → Conv2D(64) → MaxPool → Dropout | 8×8×64 |
| Block 3 | Conv2D(128) → MaxPool → Dropout | 4×4×128 |
| Head | Flatten → Dense(128) → Dropout → Dense(10, softmax) | 10 |

- All conv layers use **3×3 kernels**, ReLU activation, `same` padding
- ~65% of the weights sit in the single `Dense(128)` layer after `Flatten`

**Training config:** Adam optimizer · categorical cross-entropy · batch size 64 · 100 epochs.

---

## Dataset

**CIFAR-10** (loaded via `keras.datasets.cifar10`) — 60,000 32×32 RGB images
across 10 balanced classes (airplane, automobile, bird, cat, deer, dog, frog,
horse, ship, truck), split 50,000 train / 10,000 test. No download step is
needed; Keras fetches and caches it on first run.

---

## Getting Started

1. Clone the repo and move into this folder:

   ```bash
   git clone https://github.com/nzoopxw/deep-learning-lab-sem5.git
   cd deep-learning-lab-sem5/Lab-03
   ```

2. (Recommended) create a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Launch the notebook and run the cells top to bottom:

   ```bash
   jupyter notebook Experiment_3.ipynb
   ```

> Training for 100 epochs is much faster on a GPU. On CPU it will still run, just
> slowly — drop the epoch count if you only want to sanity-check the pipeline.

---

## Tasks in this Lab

1. **Dataset exploration** — shapes, class distribution, sample image grid
2. **Preprocessing** — normalize pixels to [0, 1], one-hot encode labels
3. **Architecture design** — build the CNN, inspect `model.summary()`
4. **Training** — 100 epochs, logging train/val accuracy and loss
5. **Feature-map visualization** — inspect first-conv-layer activations
6. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, classification report
7. **Pooling comparison** — retrain with `AveragePooling2D` and compare

---

## Results Snapshot

| Metric | Value |
|--------|-------|
| Test accuracy | 0.8361 |
| Precision (macro) | 0.8385 |
| Recall (macro) | 0.8361 |
| F1-score (macro) | 0.8362 |

Vehicle classes were the most reliable (automobile F1 0.92, ship 0.91), while
**cat** was the weakest (F1 0.70) — most of the confusion is cat↔dog, which makes
sense given their similar shape and texture at 32×32 resolution.

---

## Tech Stack

Python · TensorFlow/Keras · NumPy · Matplotlib · Seaborn · scikit-learn

See [`requirements.txt`](./requirements.txt) for the full dependency list.
