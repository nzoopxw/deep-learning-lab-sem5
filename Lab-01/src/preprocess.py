"""
Task 1 & 3: Dataset loading, exploration and preprocessing.

Banknote Authentication Dataset (UCI ML Repository)
    Features : variance, skewness, curtosis, entropy
    Target   : class (0 = Authentic, 1 = Forged)
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Anchor the data path to this file's location, NOT the current working
# directory, so the script runs correctly from anywhere.
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "data_banknote_authentication.txt"
)

COLUMNS = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
    "class",
]

FEATURES = ["variance", "skewness", "curtosis", "entropy"]


def load_raw_dataframe():
    """Load the raw dataset as a pandas DataFrame (used for EDA plots)."""

    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    return df


def explore_dataset(df=None):
    """
    Task 1: Dataset Exploration.

    Displays the first five samples, dataset dimensions, missing value
    counts and descriptive statistics.
    """

    if df is None:
        df = load_raw_dataframe()

    print("=" * 60)
    print("TASK 1: DATASET EXPLORATION")
    print("=" * 60)

    print("\nFirst five samples:")
    print(df.head())

    print("\nDataset dimensions (rows, columns):", df.shape)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df["class"].value_counts().sort_index())

    print("\nDescriptive statistics:")
    print(df.describe())

    return df


def load_and_preprocess_data(normalize=True, test_size=0.2, random_state=42):
    """
    Task 3: Data Preprocessing.

    Normalizes all numerical features (StandardScaler) and splits the
    dataset into Training (80%) and Testing (20%) sets.

    Set normalize=False to study the effect of feature normalization
    on convergence (Additional Task 5).
    """

    df = load_raw_dataframe()

    X = df[FEATURES].values
    y = df["class"].values

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = explore_dataset()

    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)