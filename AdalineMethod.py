import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class AdalineGD(object):

    def __init__(self, rate=0.01, epochs=50):
        self.rate = rate
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost = []

        for i in range(self.epochs):

            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.rate * X.T.dot(errors)
            self.w_[0] += self.rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
        return self



    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


data = pd.read_csv('IRIS_100.csv', header=None)

y = data.iloc[0:100, 4].values
y = np.where(y == 'setosa', -1, 1)

X = data.iloc[0:100, [0, 2]].values

# Standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

adaline = AdalineGD(rate=0.01, epochs=15)

adaline.train(X_std, y)
plot_decision_regions(X_std, y, clf=adaline, hide_spines=False, colors='blue,green', legend=2)
plt.title('Adaline Gradient Descent')
plt.xlabel('Sepal Length (Standardized)')
plt.ylabel('Petal Length (Standardized)')
legend_label = plt.legend()
legend_label.get_texts()[0].set_text('Setosa')
legend_label.get_texts()[1].set_text('Versicolor')
plt.show()

