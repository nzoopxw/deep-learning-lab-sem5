"""
Experiment 1: Single Layer Perceptron for Binary Classification.

Main driver. Runs every task in the lab sheet, in order, and writes all
mandatory plots to <project_root>/outputs/plots.

Run from anywhere:
    python src/train.py
"""

import numpy as np

import visualization as viz
from preprocess import (
    FEATURES,
    explore_dataset,
    load_and_preprocess_data,
    load_raw_dataframe,
)
from perceptron import Perceptron
from evaluation import (
    compare_with_sklearn,
    evaluate_model,
    print_comparison_table,
    print_epoch_table,
    print_training_summary,
)

LEARNING_RATE = 0.01
EPOCHS = 50
LEARNING_RATES = [0.001, 0.01, 0.1]


def main():

    # ------------------------------------------------------------------
    # Task 1: Dataset exploration
    # ------------------------------------------------------------------
    df = load_raw_dataframe()
    explore_dataset(df)

    # ------------------------------------------------------------------
    # Task 2: Exploratory Data Analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    viz.plot_feature_histograms(df)
    viz.plot_correlation_heatmap(df)
    viz.plot_scatterplots(df)
    viz.plot_boxplots(df)

    # ------------------------------------------------------------------
    # Task 3: Preprocessing
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("\n" + "=" * 60)
    print("TASK 3: DATA PREPROCESSING")
    print("=" * 60)
    print("Features normalized with StandardScaler (zero mean, unit variance).")
    print("Training set shape:", X_train.shape)
    print("Test set shape    :", X_test.shape)

    # ------------------------------------------------------------------
    # Task 4 & 5: Perceptron implementation and training
    # ------------------------------------------------------------------
    model = Perceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS, verbose=True)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Task 6: Evaluation
    # ------------------------------------------------------------------
    metrics = evaluate_model(model, X_test, y_test)

    print_training_summary(
        model,
        metrics,
        n_samples=len(df),
        train_size=len(X_train),
        test_size=len(X_test),
    )
    print_epoch_table(model)

    # ------------------------------------------------------------------
    # Mandatory plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING MANDATORY PLOTS")
    print("=" * 60)

    viz.plot_training_error(model)
    viz.plot_weight_evolution(model, feature_names=FEATURES)
    viz.plot_bias_evolution(model)
    viz.plot_confusion_matrix(model, X_test, y_test)

    # ------------------------------------------------------------------
    # Task 7 / Additional Task 3: learning rate study
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 7: LEARNING RATE ANALYSIS")
    print("=" * 60)

    histories = []

    for lr in LEARNING_RATES:
        m = Perceptron(learning_rate=lr, epochs=EPOCHS, verbose=False)
        m.fit(X_train, y_train)
        histories.append(m.errors)

        lr_metrics = evaluate_model(
            m, X_test, y_test, title=f"Evaluation (eta = {lr})"
        )

        converged = (
            f"epoch {m.converged_epoch}" if m.converged_epoch else "not reached"
        )
        print(f"\n  eta = {lr:<6} | final errors = {m.errors[-1]:<4} | "
              f"convergence = {converged} | "
              f"accuracy = {lr_metrics['accuracy']:.4f}")

    viz.plot_learning_rate_comparison(histories, LEARNING_RATES)

    # ------------------------------------------------------------------
    # Additional Task 1: Step vs Sigmoid
    # ------------------------------------------------------------------
    viz.plot_activation_comparison()

    # ------------------------------------------------------------------
    # Additional Task 2: Scikit-learn comparison
    # ------------------------------------------------------------------
    sklearn_metrics = compare_with_sklearn(
        X_train, X_test, y_train, y_test,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
    )
    print_comparison_table(metrics, sklearn_metrics)

    # ------------------------------------------------------------------
    # Additional Task 4: XOR
    # ------------------------------------------------------------------
    viz.plot_xor_problem()

    xor_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    xor_y = np.array([0, 1, 1, 0])

    xor_model = Perceptron(learning_rate=0.1, epochs=20, verbose=False)
    xor_model.fit(xor_X, xor_y)

    print("\n" + "=" * 60)
    print("ADDITIONAL TASK 4: XOR PROBLEM")
    print("=" * 60)
    print(f"Errors per epoch on XOR: {xor_model.errors}")
    print("The error count never reaches zero: XOR is not linearly separable,")
    print("so no single hyperplane w^T x + b = 0 can separate the classes.")

    # ------------------------------------------------------------------
    # Additional Task 5: effect of normalization
    # ------------------------------------------------------------------
    Xr_train, Xr_test, yr_train, yr_test = load_and_preprocess_data(
        normalize=False
    )

    raw_model = Perceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS,
                           verbose=False)
    raw_model.fit(Xr_train, yr_train)

    raw_metrics = evaluate_model(
        raw_model, Xr_test, yr_test, title="Evaluation (unnormalized features)"
    )

    print("\n" + "=" * 60)
    print("ADDITIONAL TASK 5: EFFECT OF NORMALIZATION")
    print("=" * 60)
    print(f"Normalized   -> final errors = {model.errors[-1]:<4} "
          f"accuracy = {metrics['accuracy']:.4f}")
    print(f"Unnormalized -> final errors = {raw_model.errors[-1]:<4} "
          f"accuracy = {raw_metrics['accuracy']:.4f}")

    viz.plot_normalization_comparison(model.errors, raw_model.errors)

    # ------------------------------------------------------------------
    # Optional: decision boundary on two features
    # ------------------------------------------------------------------
    two_feature_model = Perceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS,
                                   verbose=False)
    two_feature_model.fit(X_train[:, :2], y_train)

    viz.plot_decision_boundary(
        two_feature_model,
        X_train[:, :2],
        y_train,
        feature_names=(FEATURES[0], FEATURES[1]),
    )

    print("\n" + "=" * 60)
    print(f"All plots saved to: {viz.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()