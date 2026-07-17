"""
All plotting for Experiment 1.

Every figure is saved to <project_root>/outputs/plots at 600 dpi with
Times New Roman at font size 15, and carries a title, x label, y label
and a legend.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Anchor to the project root, NOT the current working directory.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 600

FEATURES = ["variance", "skewness", "curtosis", "entropy"]

CLASS_LABELS = {0: "Authentic (0)", 1: "Forged (1)"}

sns.set_style("whitegrid")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 15,
    "figure.titlesize": 15,
})


def _save(fig, filename):
    """Save a figure to OUTPUT_DIR and report the absolute path."""

    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ----------------------------------------------------------------------
# Task 2: Feature histograms
# ----------------------------------------------------------------------

def plot_feature_histograms(df):

    colours = ["#3E92CC", "#E76F51", "#43AA8B", "#9D4EDD"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, feature, colour in zip(axes.ravel(), FEATURES, colours):

        sns.histplot(
            df[feature],
            bins=25,
            kde=True,
            color=colour,
            edgecolor="black",
            alpha=0.8,
            ax=ax,
            label=feature.capitalize(),
        )

        ax.set_title(f"Distribution of {feature.capitalize()}")
        ax.set_xlabel(f"{feature.capitalize()} Value")
        ax.set_ylabel("Frequency")
        ax.legend(title="Feature", loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Feature Distributions (Banknote Authentication Dataset)")
    fig.tight_layout()
    _save(fig, "feature_histograms.png")


# ----------------------------------------------------------------------
# Task 2: Correlation heatmap
# ----------------------------------------------------------------------

def plot_correlation_heatmap(df):

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="vlag",
        linewidths=0.6,
        square=True,
        fmt=".2f",
        annot_kws={"size": 15},
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")

    # A heatmap has no natural legend, so the colourbar is labelled and a
    # proxy legend records the sign convention.
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12,
                   color="#B2182B", label="Positive correlation"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12,
                   color="#2166AC", label="Negative correlation"),
    ]
    ax.legend(
        handles=handles,
        title="Interpretation",
        loc="upper left",
        bbox_to_anchor=(1.35, 1.0),
    )

    _save(fig, "feature_correlation_heatmap.png")


# ----------------------------------------------------------------------
# Task 2: Scatter plots
# ----------------------------------------------------------------------

def plot_scatterplots(df):

    pairs = list(combinations(FEATURES, 2))

    colour_pairs = [
        ("#4361EE", "#F72585"),
        ("#2A9D8F", "#E76F51"),
        ("#8338EC", "#8AC926"),
        ("#118AB2", "#FFB703"),
        ("#3A86FF", "#FB5607"),
        ("#9D4EDD", "#06D6A0"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, ((x, y), ax) in enumerate(zip(pairs, axes.ravel())):

        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue="class",
            palette={0: colour_pairs[i][0], 1: colour_pairs[i][1]},
            s=60,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            ax=ax,
        )

        ax.set_title(f"{x.title()} vs {y.title()}")
        ax.set_xlabel(f"{x.title()} Value")
        ax.set_ylabel(f"{y.title()} Value")

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=[CLASS_LABELS[0], CLASS_LABELS[1]],
            title="Class",
            loc="best",
        )
        ax.grid(alpha=0.25)

    fig.suptitle("Pairwise Feature Scatter Plots by Class")
    fig.tight_layout()
    _save(fig, "scatterplots.png")


# ----------------------------------------------------------------------
# Task 2: Boxplots
# ----------------------------------------------------------------------

def plot_boxplots(df):

    colours = ["#3E92CC", "#E76F51"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, feature in zip(axes.ravel(), FEATURES):

        sns.boxplot(
            data=df,
            x="class",
            y=feature,
            hue="class",
            palette={0: colours[0], 1: colours[1]},
            legend=False,
            ax=ax,
        )

        ax.set_title(f"{feature.capitalize()} by Class")
        ax.set_xlabel("Class")
        ax.set_ylabel(f"{feature.capitalize()} Value")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "1"])

        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=12,
                       color=colours[0], label=CLASS_LABELS[0]),
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=12,
                       color=colours[1], label=CLASS_LABELS[1]),
        ]
        ax.legend(handles=handles, title="Class", loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("Feature Boxplots by Class (Outlier Analysis)")
    fig.tight_layout()
    _save(fig, "boxplots.png")


# ----------------------------------------------------------------------
# Mandatory: Training error vs epoch
# ----------------------------------------------------------------------

def plot_training_error(model):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        range(1, len(model.errors) + 1),
        model.errors,
        marker="D",
        markersize=6,
        color="#F4A261",
        linewidth=2.5,
        label=f"Misclassified samples (eta = {model.learning_rate})",
    )

    if model.converged_epoch is not None:
        ax.axvline(
            model.converged_epoch,
            color="#2A9D8F",
            linestyle="--",
            linewidth=2,
            label=f"Convergence (epoch {model.converged_epoch})",
        )

    ax.set_title("Training Error vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of Misclassified Samples")
    ax.legend(title="Legend", loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "training_error.png")


# ----------------------------------------------------------------------
# Mandatory: Weight evolution
# ----------------------------------------------------------------------

def plot_weight_evolution(model, feature_names=None):

    weights = np.array(model.weight_history)

    if feature_names is None:
        feature_names = FEATURES

    weight_colours = ["#E63946", "#457B9D", "#2A9D8F", "#FF9F1C"]

    fig, ax = plt.subplots(figsize=(9, 6))

    for i in range(weights.shape[1]):

        ax.plot(
            range(1, weights.shape[0] + 1),
            weights[:, i],
            label=f"W{i + 1} ({feature_names[i]})",
            color=weight_colours[i % len(weight_colours)],
            linewidth=2,
        )

    ax.set_title("Weight Evolution Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
    ax.legend(title="Weight", loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "weight_evolution.png")


# ----------------------------------------------------------------------
# Mandatory: Bias evolution
# ----------------------------------------------------------------------

def plot_bias_evolution(model):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        range(1, len(model.bias_history) + 1),
        model.bias_history,
        marker="s",
        markersize=6,
        color="#9D4EDD",
        linewidth=2,
        label=f"Bias (eta = {model.learning_rate})",
    )

    ax.set_title("Bias Evolution Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bias Value")
    ax.legend(title="Parameter", loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "bias_evolution.png")


# ----------------------------------------------------------------------
# Mandatory: Confusion matrix
# ----------------------------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test, filename="confusion_matrix.png"):

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[CLASS_LABELS[0], CLASS_LABELS[1]],
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))

    disp.plot(cmap="Purples", colorbar=True, ax=ax)

    for text in disp.text_.ravel():
        text.set_fontsize(15)

    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    tn, fp, fn, tp = cm.ravel()

    handles = [
        plt.Line2D([0], [0], linestyle="", label=f"TN = {tn}"),
        plt.Line2D([0], [0], linestyle="", label=f"FP = {fp}"),
        plt.Line2D([0], [0], linestyle="", label=f"FN = {fn}"),
        plt.Line2D([0], [0], linestyle="", label=f"TP = {tp}"),
    ]
    ax.legend(
        handles=handles,
        title="Counts",
        loc="upper left",
        bbox_to_anchor=(1.35, 1.0),
        handlelength=0,
    )

    _save(fig, filename)


# ----------------------------------------------------------------------
# Task 7: Learning rate comparison
# ----------------------------------------------------------------------

def plot_learning_rate_comparison(histories, learning_rates):

    lr_colours = ["#FF006E", "#3A86FF", "#06D6A0", "#FFBE0B", "#8338EC"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (history, lr) in enumerate(zip(histories, learning_rates)):

        ax.plot(
            range(1, len(history) + 1),
            history,
            marker="o",
            linewidth=2,
            markersize=5,
            color=lr_colours[i % len(lr_colours)],
            label=f"eta = {lr}",
        )

    ax.set_title("Learning Rate Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of Misclassified Samples")
    ax.legend(title="Learning Rate", loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "learning_rate_comparison.png")


# ----------------------------------------------------------------------
# Additional Task 5: Effect of normalization
# ----------------------------------------------------------------------

def plot_normalization_comparison(errors_scaled, errors_raw):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        range(1, len(errors_scaled) + 1),
        errors_scaled,
        marker="o",
        linewidth=2,
        markersize=5,
        color="#2A9D8F",
        label="Normalized features",
    )

    ax.plot(
        range(1, len(errors_raw) + 1),
        errors_raw,
        marker="^",
        linewidth=2,
        markersize=5,
        color="#E63946",
        label="Raw (unnormalized) features",
    )

    ax.set_title("Effect of Feature Normalization on Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of Misclassified Samples")
    ax.legend(title="Preprocessing", loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "normalization_comparison.png")


# ----------------------------------------------------------------------
# Additional Task 1: Step vs Sigmoid activation
# ----------------------------------------------------------------------

def plot_activation_comparison():

    z = np.linspace(-8, 8, 400)

    step = np.where(z >= 0, 1.0, 0.0)
    sigmoid = 1.0 / (1.0 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(z, step, color="#E63946", linewidth=2.5,
            label="Step: f(z) = 1 if z >= 0 else 0")
    ax.plot(z, sigmoid, color="#3A86FF", linewidth=2.5,
            label="Sigmoid: f(z) = 1 / (1 + e^-z)")

    ax.set_title("Step vs Sigmoid Activation Function")
    ax.set_xlabel("Weighted Sum z")
    ax.set_ylabel("Activation Output f(z)")
    ax.legend(title="Activation", loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "activation_comparison.png")


# ----------------------------------------------------------------------
# Optional: Decision boundary
# ----------------------------------------------------------------------

def plot_decision_boundary(model, X, y, feature_names=("variance", "skewness")):
    """
    Optional plot. `model` must be trained on exactly the two features
    contained in X (an n x 2 array).
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.contourf(xx, yy, zz, alpha=0.2, cmap="coolwarm", levels=1)

    for cls, colour in zip([0, 1], ["#4361EE", "#F72585"]):
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colour,
            s=40,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label=CLASS_LABELS[cls],
        )

    # Separating hyperplane: w1*x + w2*y + b = 0
    w1, w2 = model.weights
    b = model.bias

    if abs(w2) > 1e-12:
        xs = np.array([x_min, x_max])
        ys = -(w1 * xs + b) / w2
        ax.plot(xs, ys, color="black", linewidth=2.5,
                label="Decision boundary")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_title("Perceptron Decision Boundary "
                 f"({feature_names[0].title()} vs {feature_names[1].title()})")
    ax.set_xlabel(f"{feature_names[0].title()} (normalized)")
    ax.set_ylabel(f"{feature_names[1].title()} (normalized)")
    ax.legend(title="Legend", loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "decision_boundary.png")


# ----------------------------------------------------------------------
# Additional Task 4: XOR problem
# ----------------------------------------------------------------------

def plot_xor_problem():

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    fig, ax = plt.subplots(figsize=(7, 6))

    for cls, colour, marker in zip([0, 1], ["#4361EE", "#F72585"], ["o", "s"]):
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colour,
            s=350,
            marker=marker,
            edgecolor="black",
            linewidth=1.0,
            label=f"XOR output = {cls}",
        )

    ax.set_title("The XOR Problem: Not Linearly Separable")
    ax.set_xlabel("Input x1")
    ax.set_ylabel("Input x2")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.35, 1.35)
    ax.legend(title="Class", loc="center")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "xor_problem.png")