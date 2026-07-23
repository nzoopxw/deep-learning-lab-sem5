# Lab 01 — Single Layer Perceptron for Binary Classification

CS3807 – Deep Learning Laboratory · Experiment 1

Implementation of a Single Layer Perceptron **from scratch** (NumPy) for binary
classification on the Banknote Authentication dataset, covering exploratory data
analysis, preprocessing, the perceptron learning rule, evaluation, and a study of
learning-rate effects on convergence.

## Dataset

**Banknote Authentication Dataset** — UCI Machine Learning Repository.
1,372 instances, 4 numerical features (Variance, Skewness, Curtosis, Entropy),
2 classes (0 = authentic, 1 = forged), no missing values. Loaded directly from
the UCI repository.

Source: https://archive.ics.uci.edu/dataset/267/banknote+authentication

## Repository Contents

| File | Description |
|------|-------------|
| `dl_lab_01.ipynb` | Full notebook: EDA, preprocessing, from-scratch perceptron, training, evaluation, learning-rate study, and all plots |
| `requirements.txt` | Python dependencies |
| `plots/` | Generated figures (histograms, heatmap, scatter, training error, weight/bias evolution, confusion matrix, learning-rate comparison, decision boundary) |
| `README.md` | This file |

## Model

A Single Layer Perceptron implemented from scratch with zero-initialized weights
and bias, a step activation function, forward propagation, and the perceptron
learning rule `w <- w + lr * (y - y_pred) * x`. Features are standardized and the
data is split 80/20 into training and test sets.

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

## Execution Instructions

**Option A — Google Colab (recommended):**
1. Open `dl_lab_01.ipynb` in Google Colab.
2. Run all cells top to bottom (*Runtime → Run all*). The dataset is fetched
   automatically from the UCI repository, so no manual download is needed.
3. Generated plots are saved to the `plots/` folder during execution.

**Option B — Local:**
1. Create and activate a virtual environment, then `pip install -r requirements.txt`.
2. Launch Jupyter and open the notebook: `jupyter notebook dl_lab_01.ipynb`.
3. Run all cells in order.

## Reproducibility

Random seeds are fixed (`numpy` set to 42) and the train/test split uses
`random_state=42`, so results are reproducible across runs.

## Notes

- The perceptron uses only the step activation function, as specified for this
  experiment.
- Precision, recall, and F1 are reported for the positive (forged) class.
- The learning-rate study repeats training at 0.001, 0.01, and 0.1 to compare
  convergence behaviour.
