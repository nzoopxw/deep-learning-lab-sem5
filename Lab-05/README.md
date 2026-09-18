# Lab 05 — CNN Training, Regularization, Optimization & Cross-Validation

A comprehensive study of the design choices that drive CNN performance —
weight initialization, regularization, batch normalization, optimizers,
hyperparameter tuning, transfer learning, fine-tuning, and **5-fold
cross-validation** for model selection. Built with **TensorFlow/Keras** using
**MobileNetV2** on the **Oxford-IIIT Pet** dataset. This is Experiment 5 of the
CS3807 Deep Learning Laboratory (Semester V, AI & DS).

---

## What This Lab Covers

Each factor is studied in isolation (change one thing at a time), with
training/validation curves, and the promising configurations are then compared
under proper cross-validation before a final held-out test.

1. **Weight initialization** — zeros, random, Xavier/Glorot, He
2. **Regularization & overfitting** — none, L2, Dropout, Batch Norm
3. **Batch Normalization** — with vs. without (plus a worked numerical example)
4. **Optimizers** — SGD, Momentum, RMSProp, Adam
5. **Hyperparameter tuning** — learning rate, batch size, dropout rate
6. **Transfer learning** — feature extraction vs. fine-tuning
7. **5-fold cross-validation** — model selection on mean ± SD
8. **Final evaluation** — retrain the winner, test once on the untouched set

---

## Model & Dataset

- **Backbone:** MobileNetV2 pretrained on ImageNet (lightweight, CPU-friendly — depthwise separable convs, inverted residuals, linear bottlenecks, ReLU6)
- **Dataset:** Oxford-IIIT Pet — 37 cat/dog breeds, RGB images resized to **224×224×3**
- **Setup:** separate train / validation / test splits; the **test set stays untouched** until final evaluation

The final model uses a frozen MobileNetV2 base with a small trainable classifier
head (only ~47K trainable parameters), so training is very cheap.

---

## Getting Started

1. Clone the repo and move into this folder:

   ```bash
   git clone https://github.com/nzoopxw/deep-learning-lab-sem5.git
   cd deep-learning-lab-sem5/Lab-05
   ```

2. (Recommended) create a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Launch the notebook and run the cells top to bottom:

   ```bash
   jupyter notebook Experiment_5.ipynb
   ```

> The Oxford-IIIT Pet dataset is loaded via `tensorflow-datasets` and downloads
> (~800 MB) on first run, then caches locally. MobileNetV2 also fetches its
> ImageNet weights on first use.

---

## Key Findings

- **Initialization:** all four converge to ~81–83% val accuracy — a single dense
  head has no hidden layer, so even zero-init breaks symmetry here (it only
  fails in deeper nets).
- **Regularization:** the plain model overfits (train ~96% vs. val ~76%); L2 /
  Dropout / BN close the generalization gap.
- **Optimizers:** RMSProp (81.6%) and Adam (79.9%) converge in ~2 epochs;
  Momentum lags (~73%); plain SGD stalls (~12%) at the same LR.
- **Best single settings:** LR `1e-3`, batch 64, dropout 0.5 (classic inverted-U on each).
- **Transfer learning:** feature extraction hits ~80% fast; fine-tuning at `1e-5`
  crawls to ~62% in the same budget — the pretrained features already transfer
  strongly, so feature extraction wins on both accuracy and cost here.

---

## Cross-Validation & Final Model

Four configurations compared with 5-fold CV (mean ± SD):

| Config | Description | CV Accuracy (%) |
|--------|-------------|-----------------|
| C1 | Adam, Dropout 0.5 | 91.20 ± 0.50 |
| C2 | Adam, BatchNorm | 90.79 ± 0.60 |
| **C3** | **Adam, L2 (1e-4)** | **91.49 ± 0.74** |
| C4 | Adam, plain | 91.33 ± 0.33 |

**Selected: C3 (Adam + L2)** — retrained on the full train/val pool and tested once:

| Metric | Value |
|--------|-------|
| Mean CV accuracy | 91.49% |
| CV standard deviation | 0.74 |
| Test accuracy | 88.53% |
| Precision (macro) | 0.888 |
| Recall (macro) | 0.885 |
| F1-score (macro) | 0.883 |
| Trainable parameters | 47,397 |

Misclassifications concentrate among visually similar breeds (overlapping coat
colour, face shape, size). The ~3% CV-to-test gap indicates healthy
generalization.

> **Additional exercise note:** a follow-up sweep found **N1** (LR `1e-3`,
> dropout 0.5, batch 64) actually generalized best — matching C3's CV mean with
> the lowest variance (±0.45) *and* the highest test accuracy (90.38%). A good
> reminder that selection should weigh stability and held-out performance, not
> the CV mean alone.

---

## Tech Stack

Python · TensorFlow/Keras (MobileNetV2) · TensorFlow Datasets (Oxford-IIIT Pet) · NumPy · Matplotlib · Seaborn · scikit-learn (KFold, metrics)

See [`requirements.txt`](./requirements.txt) for the full dependency list.
