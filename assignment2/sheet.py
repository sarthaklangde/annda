import numpy as np
import matplotlib.pyplot as plt


def create_dataset(should_plot=False):
    input_points = np.arange(start=0, stop=np.pi, step=0.1).T
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

def manual_gaussian(X, sigma=0.1):
    y = np.empty(shape=(len(X), 1))
    for i in range(len(X)):
        y[i] = np.exp(-(X[i] - ) / (2 * (sigma ** 2)))

def gaussian_radial(X):



def rbf():
    print("Hello")
