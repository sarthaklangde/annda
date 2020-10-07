import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


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

    def fit_delta(self, X_train, y_train, lr=0.01, epochs=100, stop_tol = 1e-4):
        losses = []
        for i in range(epochs):
            # Calculate f(xk) with random weight values
            phi, y_pred = self.forward(X_train)

            # Calculate error between current values and expected values
            delta_w = (lr * ((y_train - y_pred).T @ phi)).T

            # Calculate weights
            self.weights = self.weights + delta_w

            # Calculate loss
            _, y_pred = self.forward(X_train)
            current_loss = mean_squared_error(y_train, y_pred)
            if i > 0 and np.abs(current_loss - losses[-1]) < stop_tol:
                losses.append(current_loss)
                break
            else:
                losses.append(current_loss)

        print("Total epochs: ", len(losses))
        print("Final loss: ", losses[-1])
        return losses

    def fit_delta_online(self, X_train, y_train, lr=0.01, epochs=100, stop_tol=1e-4):
        losses = []
        for i in range(epochs):
            idx = np.arange(len(X_train))
            np.random.shuffle(idx)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            for j in range(len(X_train)):
                current_x = X_shuffled[j]

                # Calculate f(xk) with random weight values
                phi, y_pred = self.forward(current_x)

                # Calculate error between current values and expected values
                delta_w = (lr * ((y_shuffled[j] - y_pred).T @ phi)).T

                # Calculate weights
                self.weights = self.weights + delta_w

            # Calculate loss
            _, y_pred = self.forward(X_train)
            current_loss = mean_squared_error(y_train, y_pred)
            if i > 0 and np.abs(current_loss - losses[-1]) < stop_tol:
                losses.append(current_loss)
                break
            else:
                losses.append(current_loss)

        print("Total epochs: ", len(losses))
        print("Final loss: ", losses[-1])
        return losses


def residual_error(y1, y2):
    err = np.abs(y1 - y2).mean()
    return err


def mean_squared_error(y1, y2):
    err = np.square(y1 - y2).mean()
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

    # print(len(pos), pos)
    model = RBF(pos=pos)
    model.fit_delta(sin_data['X_train'], sin_data['y_train'], lr=0.01, epochs=1000)
    y_pred = model.predict(sin_data['X_test'])

    y_test = sin_data['y_test'].reshape(-1, 1)

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


def main2():
    dataset = create_dataset(should_plot=False, add_noise=True)
    sin_data = dataset['sin']
    square_data = dataset['square']

    # rbf_widths = [0.1, 0.5, 0.7, 1, 1.5]
    # num_of_means = [3, 4, 5, 6, 7, 8, 9, 10]
    rbf_widths = [0.5]
    num_of_means = [6]
    lrs = [0.01, 0.02, 0.05, 0.1,  0.2]
    fig = plt.figure(figsize=plt.figaspect(0.2))
    rbf_iter = 0
    mean_iter = 0
    for lr_iter in range(len(lrs)):
        # for mean_iter in range(len(num_of_means)):
        # for rbf_iter in range(len(rbf_widths)):
        ax = fig.add_subplot(1, 5, lr_iter+1)
        means = np.linspace(0, 2 * np.pi, num_of_means[mean_iter])
        means = means[1:-1]
        pos = []
        for i in range(len(means)):
            pos.append([means[i], rbf_widths[rbf_iter]])

        model = RBF(pos=pos)

        # losses = model.fit_delta_online(sin_data['X_train'], sin_data['y_train'], lr=lrs[lr_iter], epochs=1000)
        losses = model.fit_delta(sin_data['X_train'], sin_data['y_train'], lr=lrs[lr_iter], epochs=1000)
        final_epochs = len(losses)
        mse = losses[-1]
        y_pred = model.predict(sin_data['X_test'])

        y_test = sin_data['y_test'].reshape(-1, 1)
        print("Residual Error:", residual_error(y_test, y_pred))

        ax.plot(sin_data['X_train'], sin_data['y_train'], label="sin(2x) with noise")
        ax.plot(sin_data['X_test'], y_pred, label="Prediction")
        ax.set_title("lr:{}, ep:{}, mse:{:.2} ".format(lrs[lr_iter], final_epochs, mse))
        if lr_iter == 4:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower left')

    fig.suptitle("Batch Learning with mean:{}, RBFs:{}".format(num_of_means[mean_iter]-2, rbf_widths[rbf_iter]), y=1)
    fig.tight_layout()
    plt.show()


def main2_square():
    dataset = create_dataset(should_plot=False, add_noise=True)
    sin_data = dataset['sin']
    square_data = dataset['square']

    rbf_widths = [0.1, 0.2, 0.3, 0.5, 0.7]
    fig = plt.figure(figsize=plt.figaspect(0.2))
    for rbf_iter in range(len(rbf_widths)):
        ax = fig.add_subplot(1, 5, rbf_iter + 1)
        means = np.arange(start=np.pi / 4, stop=2 * np.pi, step=np.pi / 16)
        pos = []
        for i in range(len(means)):
            pos.append([means[i], rbf_widths[rbf_iter]])

        model = RBF(pos=pos)

        losses = model.fit_delta(sin_data['X_train'], sin_data['y_train'], lr=0.01, epochs=1000)
        # losses = model.fit_delta_online(sin_data['X_train'], sin_data['y_train'], lr=0.01, epochs=1000)

        final_epochs = len(losses)
        mse = losses[-1]
        y_pred = model.predict(square_data['X_test'])

        y_test = sin_data['y_test'].reshape(-1, 1)
        print("Residual Error:", residual_error(y_test, y_pred))

        ax.plot(square_data['X_train'], square_data['y_train'], label="sin(2x) with noise")
        ax.plot(square_data['X_test'], y_pred, label="Prediction")
        ax.set_title("σ:{}, ep:{}, mse:{:.2f}".format(rbf_widths[rbf_iter], final_epochs, mse))
        if rbf_iter == 4:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower left')

    fig.suptitle("Batch Learning", y=1)
    fig.tight_layout()

    plt.show()


def main3_square():
    dataset = create_dataset(should_plot=False, add_noise=True)
    sin_data = dataset['sin']
    square_data = dataset['square']

    rbf_widths = [0.1, 0.2, 0.3, 0.5, 0.7]
    num_of_means = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    # rbf_widths = [0.5]
    # num_of_means = [6]
    # lrs = [0.01, 0.02, 0.05, 0.1, 0.2]
    # rbf_iter = 0
    # mean_iter = 0
    for mean_iter in range(len(num_of_means)):
        fig = plt.figure(figsize=plt.figaspect(0.2))
        for rbf_iter in range(len(rbf_widths)):
            ax = fig.add_subplot(1, 5, rbf_iter + 1)
            means = np.linspace(0, 2 * np.pi, num_of_means[mean_iter])
            means = means[1:-1]
            pos = []
            for i in range(len(means)):
                pos.append([means[i], rbf_widths[rbf_iter]])

            model = RBF(pos=pos)

            # losses = model.fit_delta_online(sin_data['X_train'], sin_data['y_train'], lr=0.01, epochs=1000)
            losses = model.fit_delta(square_data['X_train'], square_data['y_train'], lr=0.01, epochs=1000)
            final_epochs = len(losses)
            mse = losses[-1]
            y_pred = model.predict(square_data['X_test'])

            y_test = sin_data['y_test'].reshape(-1, 1)
            print("Residual Error:", residual_error(y_test, y_pred))

            ax.plot(square_data['X_train'], square_data['y_train'], label="sin(2x) with noise")
            ax.plot(square_data['X_test'], y_pred, label="Prediction")
            ax.set_title("σ:{}, ep:{}, mse:{:.2f}".format(rbf_widths[rbf_iter], final_epochs, mse))
            if rbf_iter == 4:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower left')

        fig.suptitle("Batch Learning with RBFs:{}".format(num_of_means[mean_iter] - 2), y=1)
        fig.tight_layout()
        plt.show()


def main4_square():
    dataset = create_dataset(should_plot=False, add_noise=True)
    sin_data = dataset['sin']
    square_data = dataset['square']

    # rbf_widths = [0.1, 0.5, 0.7, 1, 1.5]
    # num_of_means = [3, 4, 5, 6, 7, 8, 9, 10]
    rbf_widths = [0.3]
    num_of_means = [12]
    lrs = [0.01, 0.05, 0.07, 0.1, 0.2]
    fig = plt.figure(figsize=plt.figaspect(0.2))
    rbf_iter = 0
    mean_iter = 0
    for lr_iter in range(len(lrs)):
        # for mean_iter in range(len(num_of_means)):
        # for rbf_iter in range(len(rbf_widths)):
        ax = fig.add_subplot(1, 5, lr_iter + 1)
        means = np.linspace(0, 2 * np.pi, num_of_means[mean_iter])
        # means = means[1:-1]
        pos = []
        for i in range(len(means)):
            pos.append([means[i], rbf_widths[rbf_iter]])

        model = RBF(pos=pos)

        losses = model.fit_delta_online(sin_data['X_train'], sin_data['y_train'], lr=lrs[lr_iter], epochs=1000)
        # losses = model.fit_delta(square_data['X_train'], square_data['y_train'], lr=lrs[lr_iter], epochs=1000)
        final_epochs = len(losses)
        mse = losses[-1]
        y_pred = model.predict(square_data['X_test'])

        y_test = square_data['y_test'].reshape(-1, 1)
        print("Residual Error:", residual_error(y_test, y_pred))

        ax.plot(square_data['X_train'], square_data['y_train'], label="sin(2x) with noise")
        ax.plot(square_data['X_test'], y_pred, label="Prediction")
        ax.set_title("lr:{}, ep:{}, mse:{:.2} ".format(lrs[lr_iter], final_epochs, mse))
        if lr_iter == 4:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower left')

    fig.suptitle("Online Learning with RBFs:{}, σ:{}".format(num_of_means[mean_iter], rbf_widths[rbf_iter]), y=1)
    fig.tight_layout()
    plt.show()


def get_random_means(num):
    data = np.linspace(0, 2*np.pi, 1000)
    np.random.shuffle(data)
    selected_random = data[:num]
    return selected_random


def compare_rbf_position():
    dataset = create_dataset(should_plot=False, add_noise=True)
    sin_data = dataset['sin']
    square_data = dataset['square']

    # rbf_widths = [0.1, 0.5, 0.7, 1, 1.5]
    # num_of_means = [3, 4, 5, 6, 7, 8, 9, 10]
    rbf_widths = [0.3]
    num_of_means = [12]
    fig = plt.figure(figsize=plt.figaspect(0.2))
    rbf_iter = 0
    mean_iter = 0

    # means = np.linspace(0, 2 * np.pi, num_of_means[mean_iter])
    means = get_random_means(num_of_means[mean_iter])
    pos = []
    for i in range(len(means)):
        pos.append([means[i], rbf_widths[rbf_iter]])


    model = RBF(pos=pos)

    # losses = model.fit_delta_online(sin_data['X_train'], sin_data['y_train'], lr=lrs[lr_iter], epochs=1000)
    losses = model.fit_delta(square_data['X_train'], square_data['y_train'], lr=0.01, epochs=1000)
    final_epochs = len(losses)
    mse = losses[-1]
    y_pred = model.predict(square_data['X_test'])

    y_test = square_data['y_test'].reshape(-1, 1)
    print("Residual Error:", residual_error(y_test, y_pred))

    plt.plot(square_data['X_train'], square_data['y_train'], label="sin(2x) with noise")
    plt.plot(square_data['X_test'], y_pred, label="Prediction")
    plt.title("Random distribution RBFs: {}, ep:{}, mse:{:.2} ".format(num_of_means[mean_iter], final_epochs, mse))
    # fig.suptitle("Symmetric distribution of {} RBFs".format(rbf_widths[rbf_iter]), y=1)
    # fig.tight_layout()
    plt.legend()
    plt.show()


def compare_noise_free():
    dataset = create_dataset(should_plot=False, add_noise=True)
    test_dataset = create_dataset(should_plot=False, add_noise=False)
    sin_data = dataset['sin']
    square_data = dataset['square']
    test_square_data = test_dataset['square']

    # rbf_widths = [0.1, 0.5, 0.7, 1, 1.5]
    # num_of_means = [3, 4, 5, 6, 7, 8, 9, 10]
    rbf_widths = [0.3]
    num_of_means = [12]
    fig = plt.figure(figsize=plt.figaspect(0.2))
    rbf_iter = 0
    mean_iter = 0

    means = np.linspace(0, 2 * np.pi, num_of_means[mean_iter])
    # means = get_random_means(num_of_means[mean_iter])
    pos = []
    for i in range(len(means)):
        pos.append([means[i], rbf_widths[rbf_iter]])

    model = RBF(pos=pos)

    # losses = model.fit_delta_online(sin_data['X_train'], sin_data['y_train'], lr=lrs[lr_iter], epochs=1000)
    losses = model.fit_delta(square_data['X_train'], square_data['y_train'], lr=0.01, epochs=1000)
    final_epochs = len(losses)
    mse = losses[-1]
    y_pred = model.predict(test_square_data['X_test'])

    y_test = test_square_data['y_test'].reshape(-1, 1)
    print("Residual Error:", residual_error(y_test, y_pred))

    plt.plot(square_data['X_train'], square_data['y_train'], label="square(2x) with noise")
    plt.plot(test_square_data['X_test'], y_pred, label="Prediction")
    plt.title("Noise-free test RBFs: {}, ep:{}, mse:{:.2} ".format(num_of_means[mean_iter], final_epochs, mse))
    # fig.suptitle("Symmetric distribution of {} RBFs".format(rbf_widths[rbf_iter]), y=1)
    # fig.tight_layout()
    plt.legend()
    plt.show()


def compare_regression():
    dataset = create_dataset(should_plot=False, add_noise=True)
    sin_data = dataset['sin']
    square_data = dataset['square']
    regr = MLPRegressor(hidden_layer_sizes=[6], max_iter=10000, activation="tanh", learning_rate_init=0.1).fit(sin_data['X_train'].reshape(-1, 1), sin_data['y_train'])

    y_pred = regr.predict(sin_data['X_test'].reshape(-1, 1))
    # y_test = sin_data['y_test'].reshape(-1, 1)

    mse = regr.loss_
    epochs = regr.n_iter_

    plt.plot(sin_data['X_train'], sin_data['y_train'], label="sin(2x) with noise")
    plt.plot(sin_data['X_test'], y_pred, label="Prediction")
    plt.title("MLP with hidden layer size 6, ep:{}, mse:{:.2f}".format(epochs, mse))
    plt.legend()
    plt.show()
    # rbf_widths = [0.3]
    # num_of_means = [12]
    # fig = plt.figure(figsize=plt.figaspect(0.2))
    # rbf_iter = 0
    # mean_iter = 0
    #
    # means = np.linspace(0, 2 * np.pi, num_of_means[mean_iter])
    # pos = []
    # for i in range(len(means)):
    #     pos.append([means[i], rbf_widths[rbf_iter]])
    #
    # model = RBF(pos=pos)
    #
    # # losses = model.fit_delta_online(sin_data['X_train'], sin_data['y_train'], lr=lrs[lr_iter], epochs=1000)
    # losses = model.fit_delta(square_data['X_train'], square_data['y_train'], lr=0.01, epochs=1000)
    # final_epochs = len(losses)
    # mse = losses[-1]
    # y_pred = model.predict(sin_data['X_test'])
    #
    # y_test = sin_data['y_test'].reshape(-1, 1)
    # print("Residual Error:", residual_error(y_test, y_pred))
    #
    # plt.plot(square_data['X_train'], square_data['y_train'], label="square(2x) with noise")
    # plt.plot(sin_data['X_test'], y_pred, label="Prediction")
    # plt.title("Noise-free test RBFs: {}, ep:{}, mse:{:.2} ".format(num_of_means[mean_iter], final_epochs, mse))
    # plt.legend()
    # plt.show()

calculate_num_points():
