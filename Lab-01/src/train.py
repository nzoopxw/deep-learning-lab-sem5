from preprocess import load_and_preprocess_data
from perceptron import Perceptron
from evaluation import evaluate_model

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    model = Perceptron(
        learning_rate=0.01,
        epochs=50
    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

if __name__ == "__main__":
    main()