import numpy as np
import matplotlib.pyplot as plt


def create_dataset(should_plot=False, add_noise=False):
    input_points = np.arange(start=0, stop=2*np.pi, step=0.1)
    input_points = input_points.reshape(-1, 1)
    sin_output = np.sin(2 * input_points)
    square_output = np.copy(sin_output)
    square_output[square_output >= 0] = 1
    square_output[square_output < 0] = -1

    test_points = np.arange(start=0.05, stop=2*np.pi, step=0.1).T
    sin_test_output = np.sin(2 * test_points)
    square_test_output = np.copy(sin_test_output)
    square_test_output[square_test_output >= 0] = 1
    square_test_output[square_test_output < 0] = -1

    if add_noise:
        sin_output = sin_output + np.random.normal(0, 0.1, len(sin_output)).reshape(sin_output.shape)
        square_output = square_output + np.random.normal(0, 0.1, len(square_output)).reshape(square_output.shape)
        sin_test_output = sin_test_output + np.random.normal(0, 0.1, len(sin_test_output)).reshape(sin_test_output.shape)
        square_test_output = square_test_output + np.random.normal(0, 0.1, len(square_test_output)).reshape(square_test_output.shape)

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
        self.weights = np.random.randn(self.hiddenSize, 1)
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
        y_pred = phi @ self.weights
        return y_pred

    def forward(self, X_train):
        phi = manual_gaussian(X_train, self.pos)
        y_pred = phi @ self.weights
        return phi, y_pred

    def fit_delta(self, X_train, y_train, lr=0.01, epochs=100):
        for i in range(epochs):
            # Calculate f(xk) with random weight values
            phi, y_pred = self.forward(X_train)

            # Calculate error between current values and expected values
            delta_w = (lr * ((y_train - y_pred).T @ phi)).T

            # Calculate weights
            self.weights = self.weights + delta_w


def residual_error(y1, y2):
    err = np.abs(y1 - y2).mean()
    return err


def main():
    print("Main 1")
    dataset = create_dataset(False)
    sin_data = dataset['sin']
    square_data = dataset['square']

    # pos = [[2]]
    pos = []
    means = [np.pi/4, 3*np.pi/4]
    for i in range(len(means)):
        pos.append([means[i], 0.4])

    print(len(pos), pos)
    model = RBF(pos=pos)
    model.fit_delta(sin_data['X_train'], sin_data['y_train'], lr=0.01, epochs=1000)
    y_pred = model.predict(sin_data['X_test'])

    y_test = sin_data['y_test'].reshape(-1, 1)
    for i in range(len(y_test)):
        print(y_test[i], y_pred[i])

    print("Residual Error:", residual_error(y_test, y_pred))
    plt.plot(sin_data['X_train'], sin_data['y_train'], label="sin(2x)")
    plt.plot(sin_data['X_test'], y_pred, label="RBF")
    plt.legend()
    plt.show()


def plot_noisy_data():
    data1 = create_dataset()
    data2 = create_dataset(add_noise=True)

    sindata1 = data1['sin']
    sindata2 = data2['sin']
    squaredata1 = data1['square']
    squaredata2 = data2['square']

    plt.plot(sindata1['X_train'], sindata1['y_train'], label="sin(2x)")
    plt.plot(sindata2['X_train'], sindata2['y_train'], label="sin(2x) with noise")
    plt.legend()
    plt.show()

    plt.plot(squaredata1['X_train'], squaredata1['y_train'], label="square(2x)")
    plt.plot(squaredata2['X_train'], squaredata2['y_train'], label="square(2x) with noise")
    plt.legend()
    plt.show()

