"""
Task 6: Model evaluation and the performance tables from Section 8.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, title="Model Evaluation"):
    """Compute accuracy, precision, recall, F1-score and confusion matrix."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 60)
    print(f"TASK 6: {title.upper()}")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives  (TN) = {tn}")
    print(f"  False Positives (FP) = {fp}")
    print(f"  False Negatives (FN) = {fn}")
    print(f"  True Positives  (TP) = {tp}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }


def print_training_summary(model, metrics, n_samples, train_size, test_size):
    """Section 8: Training Summary table."""

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    weight_str = ", ".join(f"{w:+.4f}" for w in model.weights)

    rows = [
        ("Dataset Size", f"{n_samples}"),
        ("Train/Test Split", f"{train_size} / {test_size}  (80% / 20%)"),
        ("Learning Rate", f"{model.learning_rate}"),
        ("Epochs", f"{model.epochs}"),
        ("Converged At Epoch",
         f"{model.converged_epoch}" if model.converged_epoch else "Not reached"),
        ("Final Weights", f"[{weight_str}]"),
        ("Final Bias", f"{model.bias:+.4f}"),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ("Precision", f"{metrics['precision']:.4f}"),
        ("Recall", f"{metrics['recall']:.4f}"),
        ("F1-score", f"{metrics['f1_score']:.4f}"),
    ]

    for label, value in rows:
        print(f"{label:<22}: {value}")


def print_epoch_table(model, max_rows=None):
    """Section 8: Epoch-wise Learning table."""

    print("\n" + "=" * 60)
    print("EPOCH-WISE LEARNING")
    print("=" * 60)

    rows = model.epoch_table()

    if max_rows is not None:
        rows = rows[:max_rows]

    n_weights = len(model.weights)

    header = f"{'Epoch':>6} {'Errors':>7}"
    for j in range(1, n_weights + 1):
        header += f" {'Weight ' + str(j):>10}"
    header += f" {'Bias':>9}"

    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['epoch']:>6} {row['errors']:>7}"
        for j in range(1, n_weights + 1):
            line += f" {row[f'weight_{j}']:>10.4f}"
        line += f" {row['bias']:>9.4f}"
        print(line)

    if max_rows is not None and len(model.errors) > max_rows:
        print(f"... ({len(model.errors) - max_rows} more epochs)")


def compare_with_sklearn(X_train, X_test, y_train, y_test,
                         learning_rate=0.01, epochs=50):
    """
    Additional Task 2: compare the from-scratch implementation with
    Scikit-learn's Perceptron.
    """

    from sklearn.linear_model import Perceptron as SklearnPerceptron

    clf = SklearnPerceptron(
        eta0=learning_rate,
        max_iter=epochs,
        tol=None,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

    print("\n" + "=" * 60)
    print("ADDITIONAL TASK 2: SCIKIT-LEARN PERCEPTRON COMPARISON")
    print("=" * 60)

    weight_str = ", ".join(f"{w:+.4f}" for w in clf.coef_[0])
    print(f"Final Weights : [{weight_str}]")
    print(f"Final Bias    : {clf.intercept_[0]:+.4f}")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1-score      : {metrics['f1_score']:.4f}")

    return metrics


def print_comparison_table(scratch_metrics, sklearn_metrics):
    """Side-by-side table: from-scratch vs Scikit-learn."""

    print("\n" + "-" * 60)
    print(f"{'Metric':<12} {'From Scratch':>15} {'Scikit-learn':>15}")
    print("-" * 60)

    for key in ["accuracy", "precision", "recall", "f1_score"]:
        print(f"{key:<12} {scratch_metrics[key]:>15.4f} "
              f"{sklearn_metrics[key]:>15.4f}")