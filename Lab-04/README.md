# Lab 04 — Transfer Learning & CNN Architecture Comparison

A comparative study of deep CNN architectures on **CIFAR-10**, centered on
**transfer learning with VGG16** (frozen base → fine-tuning) and benchmarked
against LeNet-5, AlexNet, GoogleNet and ResNet trained from scratch. Built with
**TensorFlow/Keras**. This is Experiment 4 of the CS3807 Deep Learning
Laboratory (Semester V, AI & DS).

---

## Highlights

- **VGG16 transfer learning:** 78.30% test accuracy with a frozen base, rising to **85.70%** after fine-tuning `block5` (+7.40 points)
- **14,848,586** total params — only **133,898** trainable in the frozen-head stage
- Full evaluation: precision / recall / F1, confusion matrix, classification report, misclassified-image gallery
- Hyperparameter study across learning rate, batch size, optimizer, dense width, and freezing strategy
- Head-to-head comparison of 5 architectures on the same subset

---

## The Transfer-Learning Approach

VGG16 is used as a pretrained ImageNet feature extractor:

1. Load VGG16 with ImageNet weights, no top classifier
2. Resize CIFAR-10 images to **96×96** and apply the VGG16 preprocessing function
3. Freeze the convolutional base
4. Add a new head: `GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.3) → Dense(10, softmax)`
5. Train the head for 10 epochs (frozen base)
6. **Fine-tune:** unfreeze `block5`, recompile with a lower LR (`1e-4`), train 6 more epochs

The lower fine-tuning learning rate matters — the hyperparameter study shows that
unfreezing conv layers at the high `1e-3` rate collapses accuracy to ~10% by
destroying the pretrained weights.

**Training config:** Adam · LR 1e-3 (head) / 1e-4 (fine-tune) · batch size 32 · sparse categorical cross-entropy.

---

## Dataset

**CIFAR-10** (via `keras.datasets.cifar10`) — 32×32 RGB images across 10 classes.

> **Note on this run:** a subset is used to keep training practical —
> **10,000 train / 2,000 test images.** This is controlled by a single `SUBSET`
> flag in the notebook; set it to `None` to reproduce the full 50k/10k run. All
> reported results correspond to the 10k/2k subset.

---

## Getting Started

1. Clone the repo and move into this folder:

   ```bash
   git clone https://github.com/nzoopxw/deep-learning-lab-sem5.git
   cd deep-learning-lab-sem5/Lab-04
   ```

2. (Recommended) create a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Launch the notebook and run the cells top to bottom:

   ```bash
   jupyter notebook Experiment_4.ipynb
   ```

> A GPU is strongly recommended — VGG16 at 96×96 is the heaviest model here
> (~304s to train). The from-scratch models are much lighter. First run also
> downloads the ImageNet weights for VGG16 / ResNet50.

---

## Experimental Procedure

1. **Dataset preparation** — load, normalize to [0, 1], show 10 samples, print shapes
2. **Transfer learning** — VGG16 base + new classifier head (frozen)
3. **Training** — 10 epochs with the base frozen
4. **Fine-tuning** — unfreeze `block5`, retrain at a lower LR, compare before/after
5. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, classification report

Plus a **hyperparameter study** and a **from-scratch architecture comparison**.

---

## Results Snapshot

### Fine-tuned VGG16 (transfer learning)

| Metric | Value |
|--------|-------|
| Training accuracy | 0.9741 |
| Test accuracy | 0.8570 |
| Precision (macro) | 0.8585 |
| Recall (macro) | 0.8555 |
| F1-score (macro) | 0.8555 |
| Training time | 303.7 s |

Vehicle / rigid-object classes (automobile, truck, ship, horse) score highest;
cat and dog are the weakest, mostly confused with each other at low resolution.

### Architecture comparison (same 10k/2k subset)

| Model | Params | Accuracy | Training time | Training |
|-------|--------|----------|---------------|----------|
| LeNet-5 | 83,126 | 42.60% | 11.0 s | from scratch |
| AlexNet | 3,063,690 | 61.70% | 21.9 s | from scratch |
| GoogleNet | 106,810 | 45.20% | 27.2 s | from scratch |
| ResNet50 | 307,978 | 59.85% | 22.0 s | from scratch |
| **VGG16** | 14,848,586 | **85.70%** | 303.7 s | transfer learning |

> Parameter counts for the from-scratch GoogleNet/ResNet reflect the small custom
> versions used here, not the full 6.8M / 25.6M reference architectures.

VGG16 with transfer learning wins clearly by reusing ImageNet features — at a
notably higher training-time cost.

---

## Tech Stack

Python · TensorFlow/Keras (VGG16, ResNet50, MobileNetV2 applications) · NumPy · Matplotlib · Seaborn · scikit-learn

See [`requirements.txt`](./requirements.txt) for the full dependency list.
