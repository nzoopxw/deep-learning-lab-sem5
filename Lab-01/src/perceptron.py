"""
Task 4 & 5: Single Layer Perceptron implemented from scratch.

    z    = w^T x + b
    y^   = f(z)                (step activation)
    w    = w + eta * (y - y^) * x
    b    = b + eta * (y - y^)
"""

import numpy as np


class Perceptron:
    """Single layer perceptron using the step activation function."""

    def __init__(self, learning_rate=0.01, epochs=50, verbose=True):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose

        # Weights and bias are unknown until we see the feature count.
        self.weights = None
        self.bias = None

        self.errors = []
        self.weight_history = []
        self.bias_history = []
        self.converged_epoch = None

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    @staticmethod
    def step_activation(z):
        """Step activation: returns 1 if z >= 0, else 0."""

        return 1 if z >= 0 else 0

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------

    def predict_sample(self, x):
        """Forward pass for a single sample."""

        z = np.dot(x, self.weights) + self.bias
        return self.step_activation(z)

    def predict(self, X):
        """Forward pass for a batch of samples."""

        return np.array([self.predict_sample(x) for x in X])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Train using the perceptron learning rule."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # Weight and bias initialization
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        self.errors = []
        self.weight_history = []
        self.bias_history = []
        self.converged_epoch = None

        if self.verbose:
            print("=" * 60)
            print(f"TASK 5: TRAINING  (eta = {self.learning_rate}, "
                  f"epochs = {self.epochs})")
            print("=" * 60)

        for epoch in range(self.epochs):

            errors = 0

            # x_i -> one training example, target -> its true class
            for x_i, target in zip(X, y):

                prediction = self.predict_sample(x_i)

                # If the prediction is correct, update == 0 and nothing moves.
                update = self.learning_rate * (target - prediction)

                self.weights += update * x_i
                self.bias += update

                if update != 0.0:
                    errors += 1

            self.errors.append(errors)
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

            if errors == 0 and self.converged_epoch is None:
                self.converged_epoch = epoch + 1

            if self.verbose:
                weight_str = ", ".join(f"{w:+.4f}" for w in self.weights)
                print(f"Epoch {epoch + 1:>3}/{self.epochs}  | "
                      f"Misclassified: {errors:>4}  | "
                      f"Weights: [{weight_str}]  | "
                      f"Bias: {self.bias:+.4f}")

        if self.verbose:
            if self.converged_epoch is not None:
                print(f"\nConverged at epoch {self.converged_epoch} "
                      f"(zero misclassifications).")
            else:
                print("\nDid not reach zero training errors within the "
                      "epoch budget.")

        return self

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def epoch_table(self):
        """
        Epoch-wise learning table (Section 8).

        Returns a list of dicts: epoch, errors, each weight, bias.
        """

        rows = []

        for i, (errors, w, b) in enumerate(
            zip(self.errors, self.weight_history, self.bias_history), start=1
        ):
            row = {"epoch": i, "errors": errors}

            for j, wj in enumerate(w, start=1):
                row[f"weight_{j}"] = wj

            row["bias"] = b
            rows.append(row)

        return rows