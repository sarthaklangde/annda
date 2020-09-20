import numpy as np
import matplotlib.pyplot as plt

import implementation as imp


def main():
    data = imp.create_dataset(False)["sin"]
    X = data["X_train"][:, np.newaxis]
    y = data["y_train"][:, np.newaxis]
    X_test = data["X_test"][:, np.newaxis]
    y_test = data["y_test"][:, np.newaxis]
    # mu = np.array([[np.pi / 4], [3 * np.pi / 4], [5 * np.pi / 4], [7 * np.pi / 4]])
    mu = np.random.uniform(0, 2*np.pi, [4, 1])
    C = np.array([0.47])

    model = imp.RBF(mu, C, dim=[1, 1])

    model.fit_delta_online(X, y)

    y_pred = model.predict(X_test)
    plt.scatter(mu, np.zeros(len(mu)))
    plt.scatter(model.mu, np.zeros(len(model.mu)))
    plt.plot(X_test, y_test)
    plt.plot(X_test, y_pred)
    plt.show()


if __name__ == '__main__':
    main()
