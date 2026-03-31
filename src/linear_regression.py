import numpy as np

class LinearRegressionGD:
    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, x, y):
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)

        n_samples, n_features = x.shape
        self.w = [0]*n_features
        self.b = 0

        self.losses = []

        for _ in range(self.n_iters):
            y_pred = x.dot(self.w) + self.b
            error = y_pred - y

            #gradients
            dw = (1/n_samples) * x.T.dot(error)
            db = (1/n_samples) * np.sum(error)

            #update
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

            loss = np.mean(error**2)
            self.losses.append(loss)
        
    
    def predict(self, x):
        x = np.array(x).reshape(-1, 1)
        return x.dot(self.w) + self.b