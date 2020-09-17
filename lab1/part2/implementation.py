import os

import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import tensorflow as tf
from tensorflow import keras

# For disabling GPU as it took a lot of time to add layers to the network using the GPU.
# Haven't yet figured out the reason why
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def product_dict(dicts):
    """
    # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    """
    return (dict(zip(dicts, x)) for x in it.product(*dicts.values()))


def macky_glas(X):
    """
    Generate the Macky-Glas function iteratively. Each call of the function generates a new sample.
    :param X: list of previous values
    :return: X(t+1)
    """
    x_t = X[-1]
    if len(X) < 26:
        x_t25 = 0
    else:
        x_t25 = X[-26]
    return x_t + (0.2 * x_t25) / (1 + x_t25 ** 10) - 0.1 * x_t


def generate_data(add_noise=False, sigma=0.03, plot=False):
    """
    :param add_noise: Add noise to data
    :param sigma: std. dev of Gaussian noise
    :param plot: plot the resulting dataset
    :return: generated train, val, test data
    """
    # Generating the data
    x_0 = 0.15
    length = 1500
    X = np.array([x_0])
    for i in range(length - 1):
        X = np.append(X, [macky_glas(X)])

    if add_noise:
        np.random.seed(0)
        noise = np.random.normal(0, sigma, length)
        X = X + noise

    # Splitting into learning, validation and test sets
    y_train = X[300:1100]
    y_val = X[1100:1300]
    y_test = X[1300:]

    # Creating the input of the network
    x = [[X[i - 25], X[i - 20], X[i - 15], X[i - 10], X[i - 5]] for i, _ in enumerate(X)]
    x = np.array(x)
    x_train = x[300:1100]
    x_val = x[1100:1300]
    x_test = x[1300:]

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(X)
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.savefig("timeseries.png")
        plt.show()

    return x_train, y_train, x_val, y_val, x_test, y_test, X


def create_model(layer_dim, regularization=0.0):
    """
    Create the multi layer perceptron model
    :param layer_dim: list of the dimensions of each layer
    :param regularization: regularization parameter
    :return:
    """
    layer_num = len(layer_dim) - 2
    model = keras.models.Sequential()
    for i in range(layer_num):
        model.add(keras.layers.Dense(layer_dim[i + 1], input_dim=layer_dim[i],
                                     kernel_initializer=tf.keras.initializers.GlorotUniform,
                                     kernel_regularizer=keras.regularizers.l2(regularization),
                                     bias_regularizer=keras.regularizers.l2(regularization)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dense(layer_dim[-1]))
    return model
