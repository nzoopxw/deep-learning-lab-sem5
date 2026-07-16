import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.01, epochs = 50):

        self.learning_rate = learning_rate
        self.epochs = epochs

    # initially, weights and bias don't exist because we don't know YET how many features there are
        self.weights = None
        self.bias = None

    def step_activation(self,z):

        if z >= 0:
            return 1
        return 0
    
    def predict_sample(self,x):

        # basically applying the formula here. then, according to the previous function, if z turns out to be > 0, output = 1, else output = 0
        z = np.dot(x,self.weights) + self.bias
        return self.step_activation(z)
    
    def fit(self,X,y):

        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.errors = []
        for epoch in range(self.epochs):

            errors = 0

            # every iteration x_i -> one training example, gives target - it's correct class
            for x_i, target in zip(X,y):
                prediction = self.predict_sample(x_i)

                # if prediction is correct, then: target = 1, prediction = 1, update = 0.
                update = self.learning_rate * (target - prediction)
                
                # if prediction is wrong, then weights change
                self.weights += update * x_i
                self.bias += update

                if update != 0:
                    errors += 1

            self.errors.append(errors)
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Errors: {errors}")
            print(f"Weights: {self.weights}")
            print(f"Bias: {self.bias}")
            print("-" * 40)

    def predict(self, X):

        predictions = []

        for x in X:
            predictions.append(self.predict_sample(x))

        return np.array(predictions)
    