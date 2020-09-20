import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def create_dataset(should_plot=False):
    input_points = np.arange(start=0, stop=2 * np.pi, step=0.1).T
    sin_output = np.sin(2 * input_points)
    square_output = np.copy(sin_output)
    square_output[square_output >= 0] = 1
    square_output[square_output < 0] = -1

    test_points = np.arange(start=0.05, stop=2 * np.pi, step=0.1).T
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


def pdf(X, mu, sigma):
    mu = mu[np.newaxis]
    dist = la.norm(X - mu, axis=1)
    return np.exp(-dist ** 2 / (2 * sigma ** 2))


class RBF:
    def __init__(self, mu, sigma, dim, init_CL=True):
        self.sigma = sigma
        self.mu = mu.copy()
        self.w = np.random.normal(0, 0.3, [len(mu), dim[1]])
        self.Phi = None
        self.init_CL = init_CL
        self.cr = 0.1

    def CL(self, X):
        for i in X:
            idx = (np.abs(self.mu - i)).argmin()
            v = i - self.mu[idx]
            self.mu[idx] += self.cr * v

    def forward(self, X):
        self.Phi = None
        for m in self.mu:
            phi = pdf(X, m, self.sigma)[:, np.newaxis].T
            if self.Phi is None:
                self.Phi = phi
            else:
                self.Phi = np.vstack([self.Phi, phi])

        self.Phi = self.Phi.T
        return self.Phi @ self.w

    def fit_linear(self, X, y):
        _ = self.forward(X)
        Phi = self.Phi
        a = Phi.T @ Phi
        b = Phi.T @ y
        self.w = la.solve(a, b)

    def fit_delta_online(self, X, y, lr=0.01, epochs=100, stop_tol=1e-4):
        losses = []
        for i in range(epochs):
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            if self.init_CL:
                self.CL(X_shuffled)

            for j in range(len(X)):
                current_x = X_shuffled[j]
                current_y = y_shuffled[j]

                # Calculate f(xk) with random weight values
                y_pred = self.forward(current_x)

                # Calculate error between current values and expected values
                e = current_y - y_pred
                delta_w = (lr * e @ self.Phi).T

                # Calculate weights
                self.w = self.w + delta_w

            # Calculate loss
            y_pred = self.forward(X)

            current_loss = mean_squared_error(y, y_pred)
            if i > 0 and np.abs(current_loss - losses[-1]) < stop_tol:
                losses.append(current_loss)
                break
            else:
                losses.append(current_loss)
        plt.plot(losses)
        plt.show()
        print("Total epochs: ", len(losses))
        print("Final loss: ", losses[-1])

    def predict(self, X):
        return self.forward(X)


def mean_squared_error(y1, y2):
    err = np.square(y1 - y2).mean()
    return err
