import numpy as np

class LinearRegressionGD:
    def __init__(self):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, x, y):
        n_samples, n_features = X.shape
        self.w = [0]*n_features
        self.b = 0

        for _ in range(self.n_iters):
            y_pred = X.dot(self.w) + self.b
            error = y_pred - y

            #gradients
            dw = (1/n_samples) * X.T.dot(error)
            db = (1/n_samples) * np.sum(error)

            #update
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
    
    def predict(self, X):
        return X.dot(self.w) + self.b