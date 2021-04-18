import numpy as np

# Accuracy score
def accuracy_score(yTrue, yPred):
    accuracy = np.sum(yTrue == yPred) / len(yTrue)
    return accuracy


class LogisticRegression:
    def __init__(self, n_iters=1000, learning_rate=0.01):
        self.numberOfIteration = n_iters
        self.learningRate = learning_rate
        self.weights = None;
        self.bias = None


    def fit(self, X, Y): 
        numberofSamples, numberOfFeature = X.shape
        self.weights = np.zeros(numberOfFeature)
        self.bias = 0

        # prediction of y value
        for i in range(self.numberOfIteration):
            yHat = np.dot(X, self.weights) + self.bias
            yPredicted = self.sigmoid(yHat)

        # calculation of gradients
        dW = 1 / numberofSamples * np.dot(X.T, (yPredicted - Y))
        dB = 1 / numberofSamples * np.sum(yPredicted - Y)

        # updating weights and bias
        self.weights -= self.learningRate * dW
        self.bias -= self.learningRate * dB


    # prediction
    def predict(self, X):
        yHat = np.dot(X, self.weights) + self.bias
        yPredicted = self.sigmoid(yHat)
        yOutput = [1 if i > 0.5 else 0 for i in yPredicted]
        return yOutput


    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
