import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class Perceptron(object):

    def __init__(self, rate=0.01, epochs=50):
        self.rate = rate
        self.epochs = epochs

    def train(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        print('Shape %s' % X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] = update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


data = pd.read_csv('IRIS_100.csv', header=None)

# convert data into array which contain every flower name in column 4
y = data.iloc[0:100, 4].values

# find setosa in the array, if the value inside the array is setosa than convert it to -1
# else convert it to 1.
y = np.where(y == 'setosa', -1, 1)

# Sepal length and petal length

# Get the sepal and petal from all data
X = data.iloc[0:100, [0, 2]].values

perceptron = Perceptron(epochs=10, rate=0.1)

perceptron.train(X, y)
print('Weights: %s' % perceptron.w_)
plot_decision_regions(X, y, clf=perceptron, hide_spines=False, legend=2, res=0.02, colors='green,blue')
legend_label = plt.legend()
legend_label.get_texts()[0].set_text('Setosa')
legend_label.get_texts()[1].set_text('Versicolor')
plt.title('Perceptron Method')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.show()
