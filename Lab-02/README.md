# Lab 02 — Multi-Layer Perceptron for Multi-Class Image Classification

CS3807 – Deep Learning Laboratory · Experiment 2

Implementation of a Multi-Layer Perceptron (MLP) using TensorFlow/Keras on the
Fashion-MNIST dataset, covering preprocessing, model construction, training,
evaluation, and automated hyperparameter optimization with RandomizedSearchCV.

## Dataset

**Fashion-MNIST** — 70,000 grayscale images of clothing (60,000 train / 10,000
test), each 28×28 pixels across 10 classes. Loaded directly via
`keras.datasets.fashion_mnist`.

## Repository Contents

| File | Description |
|------|-------------|
| `Experiment_2.ipynb` | Full notebook: data exploration, preprocessing, baseline MLP, evaluation, hyperparameter search, and all plots |
| `requirements.txt` | Python dependencies |
| `plots/` | Generated figures (sample images, class distribution, training curves, confusion matrices, search results, comparison) |
| `README.md` | This file |

## Model

Baseline architecture: `784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`,
compiled with Adam, categorical cross-entropy, trained for 20 epochs (batch size 32).
The optimized model is selected via 5-fold RandomizedSearchCV over layer count,
neurons, learning rate, batch size, optimizer, activation, and dropout.

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

## Execution Instructions

**Option A — Google Colab (recommended):**
1. Open `Experiment_2.ipynb` in Google Colab.
2. Set the runtime to GPU: *Runtime → Change runtime type → T4 GPU*.
3. Run all cells top to bottom (*Runtime → Run all*). The scikeras/scikit-learn
   install in the first cell requires a **session restart** before the
   hyperparameter search cell — restart when prompted, then re-run from the top.
4. Generated plots are saved to `plots/` and can be downloaded as a zip from the
   final cell.

**Option B — Local:**
1. Create and activate a virtual environment, then `pip install -r requirements.txt`.
2. Launch Jupyter and open the notebook: `jupyter notebook Experiment_2.ipynb`.
3. Run all cells in order.

## Reproducibility

Random seeds are fixed (`numpy` and `tensorflow` set to 42) and the search uses
`random_state=42`, so results are reproducible across runs on the same
environment.

## Notes

- The hyperparameter search caps epochs at 10 and subsamples 15,000 training
  images to keep 5-fold cross-validation tractable; the best configuration is
  then retrained on the full training set.
- Precision, recall, and F1 are macro-averaged across the 10 classes.

