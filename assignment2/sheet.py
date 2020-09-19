import numpy as np
import matplotlib.pyplot as plt


def create_dataset(should_plot=False):
    input_points = np.arange(start=0, stop=np.pi, step=0.1)
    input_points = input_points.reshape(-1, 1)
    sin_output = np.sin(2 * input_points)
    square_output = np.copy(sin_output)
    square_output[square_output >= 0] = 1
    square_output[square_output < 0] = -1

    test_points = np.arange(start=0.05, stop=np.pi, step=0.1).T
    sin_test_output = np.sin(2 * test_points)
    square_test_output = np.copy(sin_test_output)
    square_test_output[square_test_output >= 0] = 1
    square_test_output[square_test_output < 0] = -1

    if should_plot:
        plt.plot(input_points, sin_output, label="sin(2x)")
        plt.legend()
        plt.show()

        plt.plot(input_points, square_output, label="square(2x)")
        plt.legend()
        plt.show()

    return dict({
        "sin": dict({
            "X_train": input_points,
            "y_train": sin_output,
            "X_test": test_points,
            "y_test": sin_test_output
        }),
        "square": dict({
            "X_train": input_points,
            "y_train": square_output,
            "X_test": test_points,
            "y_test": square_test_output
        }),
    })


def manual_gaussian(X, pos):
    feature = []
    for i in range(len(pos)):
        y = np.empty(shape=(len(X)))
        for j in range(len(X)):
            num = (X[j] - pos[i][0]) ** 2
            denom = 2 * (pos[i][1] ** 2)
            y[j] = np.exp(-num/denom)
        feature.append(y)

    feature = np.array(feature).T
    return feature

class RBF:
    def __init__(self, pos=[[0.75, 0.1], [2.2, 0.1]]):
        self.hiddenSize = len(pos)
        # self.weights = np.random.randn(1, self.hiddenSize)
        self.weights = None
        self.pos = pos

    def fit_linear(self, X_train, y_train):
        N, d = X_train.shape
        # First we transform input to higher space using kernel like trick (not sure)
        phi = manual_gaussian(X_train, self.pos)
        prod = np.dot(phi.T, phi)
        a = prod + (np.eye(len(prod)) * 0)
        b = phi.T @ y_train        #
        weights = np.linalg.solve(a, b)
        self.weights = weights

    def predict(self, X_test):
        phi = manual_gaussian(X_test, self.pos)
        print("phi", phi.shape, self.weights.shape)
        y_pred = phi @ self.weights
        return y_pred


def main():
    print("Main 1")
    dataset = create_dataset(False)
    sin_data = dataset['sin']
    square_data = dataset['square']

    pos = []
    means = np.arange(start=0, stop=3.1, step=0.1)
    for i in range(len(means)):
        pos.append([means[i], 0.1])

    print(pos)
    model = RBF(pos=pos)
    model.fit_linear(sin_data['X_train'], sin_data['y_train'])
    y_pred = model.predict(sin_data['X_test'])
    print(y_pred)
    plt.plot(sin_data['X_test'], y_pred)
    plt.show()

main()