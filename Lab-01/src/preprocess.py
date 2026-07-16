import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():

    columns = [
        "variance",
        "skewness",
        "curtosis",
        "entropy",
        "class"
    ]

    df = pd.read_csv("data/data_banknote_authentication.txt", header=None, names=columns)

    X = df.drop("class", axis=1)
    y = df["class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)