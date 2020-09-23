"""
Author: Boris Bubla
04.09.2020
"""

# Imports
import numpy as np
from matplotlib import pyplot as plt


# 3.1.1 Generation of Lin Sep Data
def lin_sep_data(n=100, mA=np.array([1.5, 1.5]), mB=np.array([-1.5, -1.5]), sigmaA=0.5, sigmaB=0.5):
    """
    n = samples (must be even number)
    mA = mean of distribution A
    mB = ___________________ B
    sigmaA = std. dev of A
    sigmaB = ___________ B
    """

    # keep making data until lin sep dataset is made ; will enter infinite loop if you choose shitty mean and sigmas
    separable = False
    while not separable:
        # create data
        A = np.array([np.random.normal(mA[0],sigmaA,(n//2,1)),np.random.normal(mA[1],sigmaA,(n//2,1))]).reshape(n//2,2)
        B = np.array([np.random.normal(mB[0],sigmaB,(n//2,1)),np.random.normal(mB[1],sigmaB, (n//2,1))]).reshape(n//2,2)

        # only checks for vertical or horizontal separability
        separable = any([A[:, k].max() < B[:, k].min() or A[:, k].min() > B[:, k].max() for k in range(2)])

    plt.plot(A[:, 0], A[:, 1], 'r.', label='class A')
    plt.plot(B[:, 0], B[:, 1], 'b.', label='class B')
    plt.legend()
    plt.show()


def nonlin_sep_data(n=100, mA=np.array([0.6, 0.6]), mB=np.array([-0.6, -0.6]), sigmaA=0.5, sigmaB=0.5):
    """
    Creates 2 class lin sep dataset. Check via plot to confirm nonlinear separability, it only checks vertically and horizontally!

    Inputs:
    n = samples (must be even number)
    mA = mean of distrbution A
    mB = ___________________ B
    sigmaA = std. dev of A
    sigmaB = ___________ B

    Outputs:
    data =  vertically&horizontally nonlinear separable dataset, where last column is targets (class), rest is patterns(features)
    """

    # keep making datasets until nonlin sep dataset is made ; will enter infinite loop if you choose shitty mean and sigmas
    separable = True
    while separable:
        # create data
        A = np.array(
            [np.random.normal(mA[0], sigmaA, (n // 2, 1)), np.random.normal(mA[1], sigmaA, (n // 2, 1))]).reshape(
            n // 2, 2)
        A = np.append(A, np.ones((n // 2, 1)), axis=1)  # add targets
        B = np.array(
            [np.random.normal(mB[0], sigmaB, (n // 2, 1)), np.random.normal(mB[1], sigmaB, (n // 2, 1))]).reshape(
            n // 2, 2)
        B = np.append(B, np.ones((n // 2, 1)) * -1, axis=1)  # add targets

        # print(A)
        # print(B)

        # only checks for vertical or horizontal separability - so it could still be diagonally separable
        separable = any([A[:, k].max() < B[:, k].min() or A[:, k].min() > B[:, k].max() for k in range(2)])

    # shuffle samples
    data = np.append(A, B, axis=0)
    np.random.shuffle(data)
    # print(data)

    # visualise
    plt.plot(A[:, 0], A[:, 1], 'r.', label='class A')
    plt.plot(B[:, 0], B[:, 1], 'b.', label='class B')
    plt.title('Not linearly separable generated data 2D')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.legend()
    plt.show()

    return data


def nonlin_sep_data_part2(n=100, mA=np.array([1, 0.3]), mB=np.array([0, -0.1]), sigmaA=0.2, sigmaB=0.3):
    """
    Creates 2 class nonlin sep dataset. Check via plot to confirm nonlinear separability, it only checks vertically and horizontally!

    Inputs:
    n = samples (must be even number)
    mA = mean of distrbution A
    mB = ___________________ B
    sigmaA = std. dev of A
    sigmaB = ___________ B

    Outputs:
    data =  vertically&horizontally nonlinear separable dataset, where last column is targets (class), rest is patterns(features)
    """

    # create data
    A1_col1 = np.array([np.random.normal(-mA[0], sigmaA, (n // 4, 1))]).reshape(n // 4,
                                                                                1)  # , np.random.normal(mA[1],sigmaA,(n//4,1))])#.reshape(n//4,2)
    A1_col2 = np.array([np.random.normal(mA[1], sigmaA, (n // 4, 1))]).reshape(n // 4, 1)
    print(A1_col1.shape)
    A1 = np.append(A1_col1, A1_col2, axis=1)
    print(A1.shape)

    A2_col1 = np.array([np.random.normal(mA[0], sigmaA, (n // 4, 1))]).reshape(n // 4,
                                                                               1)  # , np.random.normal(mA[1],sigmaA,(n//4,1))])#.reshape(n//4,2)
    A2_col2 = np.array([np.random.normal(mA[1], sigmaA, (n // 4, 1))]).reshape(n // 4, 1)
    A2 = np.append(A2_col1, A2_col2, axis=1)
    # print(A1.shape)
    # A2 = np.array([np.random.normal(mA[0],sigmaA,(n//4,1)), np.random.normal(mA[1],sigmaA,(n//4,1))]).reshape(n//4,2)
    # print(A2)
    A = np.append(A1, A2, axis=0)
    # print(A.shape)
    A = np.append(A.reshape(n // 2, 2), np.ones((n // 2, 1)), axis=1)  # add targets
    # print()
    #     B = np.array([np.random.normal(mB[0],sigmaB,(n//2,1)), np.random.normal(mB[1],sigmaB,(n//2,1))]).reshape(n//2,2)

    B_col1 = np.array([np.random.normal(mB[0], sigmaB, (n // 2, 1))]).reshape(n // 2, 1)
    B_col2 = np.array([np.random.normal(mB[1], sigmaB, (n // 2, 1))]).reshape(n // 2, 1)
    B = np.append(B_col1, B_col2, axis=1)
    B = np.append(B, np.ones((n // 2, 1)) * -1, axis=1)  # add targets

    # shuffle samples
    data = np.append(A, B, axis=0)
    np.random.shuffle(data)
    # print(data)

    # visualise
    plt.plot(A[:, 0], A[:, 1], 'r.', label='class A')
    plt.plot(B[:, 0], B[:, 1], 'b.', label='class B')
    plt.title('Not linearly separable generated data 2D')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.legend()
    plt.show()

    return data


class single_layered_perceptron():
    """
    Creates a single layer perceptron with 2 inputs and 2 outputs.
    Change layers and weight&bias initialisation shapes for other input/output dimensions.

    Inputs:
    layers = [input,output] ie dxo
    update = 'batch' or 'sequential' learning
    learning = 'delta' rule or 'perceptron' rule
    lr = learning rate
    weights = init by randomly sampled around 0 with std dev 1
    bias = init as 0s
    """

    # nxo = (nx(d+1)) x ((d+1)xo)
    # Y = X x W

    def __init__(self, layers=[2, 1], lr=.005, update='batch', rule='perceptron'):

        self.layers = layers
        self.lr = lr
        self.learning_rule = rule
        self.update = update
        # dxo
        weights = np.random.normal(0, 1, (layers[0], layers[1]))
        # 1xo
        bias = np.zeros((1, layers[1]))
        # (d+1)xo
        self.weights = np.append(weights, bias, axis=0)  # add weights of bias

    def rule(self, y_hat, targets):

        if (self.learning_rule == 'perceptron'):

            # make prediction using step function
            pred = (y_hat >= 0) * 1  # 0/1 predictions
            pred = np.where(pred == 0, -1, pred)  # -1/1 predictions
            # different error calculation because single (1x1) target value creates some bugs
            if type(targets) is np.float64:
                error = targets - pred
            else:
                error = np.subtract(targets.reshape(targets.shape[0], self.layers[1]), pred)
            misclassifications = np.count_nonzero(error)

            return error, misclassifications

        if (self.learning_rule == 'delta'):

            # count misclassifications for comparison - same as preceptron learning
            pred = (y_hat >= 0) * 1  # 0/1 predictions
            pred = np.where(pred == 0, -1, pred)  # -1/1 predictions

            if type(targets) is np.float64:
                misclass = targets - pred
            else:
                misclass = np.subtract(targets.reshape(targets.shape[0], self.layers[1]), pred)
            misclassifications = np.count_nonzero(misclass)

            # error calculation for delta
            # different error calculation because single (1x1) target value creates some bugs
            if type(targets) is np.float64:
                error = targets - y_hat
                # print(error)
            else:
                error = np.subtract(targets.reshape(targets.shape[0], self.layers[1]), y_hat)
                # print(error.shape)
                # print(error)
                # print('error shape\n',error.shape)
            # calc mean squared error
            mse = (np.mean(error, axis=0)) ** 2

            return error, misclassifications

        else:  # change to raise ValError
            raise ValueError("Error - please use 'perceptron' or 'delta' as learning rule parameter.")

    def fit(self, patterns, targets, epochs):
        """
        patterns is X data (nxd)
        targets is Y labels (nx1)
        epochs is number of training iterations (int)
        """
        # n x (d+1)
        self.patterns = np.append(patterns, np.ones((patterns.shape[0], 1)), axis=1)
        self.targets = targets

        if self.update == 'batch':
            # for each bach
            performance = {'Epoch': [], 'Error Metric - Misclassifications': []}
            for i in range(epochs):
                # predict
                y_hat = np.dot(self.patterns, self.weights)

                # calc error: either delta or perceptron rule
                error, error_metric = self.rule(y_hat, self.targets)

                # store results
                performance['Epoch'].append(i + 1)
                performance['Error Metric - Misclassifications'].append(error_metric)

                # update weights
                # (d+1)xo = (d+1)xo + lr*((oxn)*nx(d+1)).T
                # print('error.T\n',error.T.shape)
                # print('patterns\n',self.patterns)
                self.weights = self.weights + self.lr * (np.dot(error.T, self.patterns)).T
                # print('WEIGHTS\n',self.weights)

            return performance, self.weights

        if self.update == 'sequential':

            performance = {'Epoch': [], 'Error Metric - Misclassifications': []}
            for i in range(epochs):
                # for each sample
                errormetric_per_epoch = []
                for j, sample in enumerate(self.patterns):
                    # predict
                    y_hat = np.dot(sample, self.weights)

                    # calc error: either delta or perceptron rule
                    error, error_metric = self.rule(y_hat, self.targets[j])
                    # print(error.shape)
                    errormetric_per_epoch.append(error_metric)

                    # update weights
                    self.weights = self.weights + self.lr * (np.dot(error.reshape(error.shape[0], self.layers[1]).T,
                                                                    sample.reshape(1, self.patterns.shape[1]))).T

                # convert to sample errors to error metric depending on rule
                if self.learning_rule == 'delta':
                    # errormetric_per_epoch = np.mean(errormetric_per_epoch)
                    errormetric_per_epoch = np.count_nonzero(errormetric_per_epoch)
                if self.learning_rule == 'perceptron':
                    errormetric_per_epoch = np.count_nonzero(errormetric_per_epoch)

                # store results
                performance['Error Metric - Misclassifications'].append(errormetric_per_epoch)
                performance['Epoch'].append(i + 1)

            return performance, self.weights

        else:
            raise ValueError("Error - please use 'perceptron' or 'delta' as learning rule parameter.")


# plots a decision boundary based on the y = formula
def decision_boundary(patterns, targets, final_weights, bias, rule, update, learning_step):
    # visualise
    #     plt.plot(A[:, 0], A[:, 1], 'r.',label='class A')
    #     plt.plot(B[:, 0], B[:, 1], 'b.',label='class B')

    #     graph(lambda x: (final_weights.T*x)+bias, range(-2, 2))
    x = np.linspace(-1, 1, 30)
    print(patterns.shape)
    y = -(final_weights[0] * x + bias) / final_weights[1]
    print(y.shape)
    print(x.shape)
    plt.plot(x, y, label='Decision boundary')
    plt.xlim(left=-2, right=2)
    plt.ylim(bottom=-2, top=2)
    # graph(lambda x: (np.dot(final_weights.T,x))+bias)

    colormap = np.array(['r', 'b'])

    red = np.where(targets == -1)
    blue = np.where(targets == 1)
    # print(red)
    # print(patterns.shape)

    plt.scatter(patterns.T[0][red], patterns.T[1][red], c='r', label='class 1', s=8)
    plt.scatter(patterns.T[0][blue], patterns.T[1][blue], c='b', label='class 2', s=8)

    # plt.plot(patterns.)
    # plt.xlabel('potential [uV] in {} at {} ms'.format(chans[0], time_points[0]))
    plt.title('{} with {} update after 25 epochs with learning step of {}'.format(rule, update, learning_step))
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.legend()
    plt.show()


#prints a plot of the learning curve - misclassifications over epochs

def learning_curve(epochs,misclassifications):
    plt.title('Learning curve for {} with {} update and a learning rate of {}'.format('delta rule','batch','rate'))
    plt.plot(performance['Epoch'],performance['Error Metric - Misclassifications'])
    plt.ylabel('Misclassifications')
    plt.xlabel('Epochs')
    plt.show()


# creates a random sampled subset of 2 class dataset data, removing a given % from each class
def random_subset(data, perc_A_removed=0, perc_B_removed=0):
    # dataA = data[np.where(data.T[2]==1)]
    dataA = data[np.where(data.T[2] == 1)]
    dataB = data[np.where(data.T[2] == -1)]
    print(dataA.shape)
    print(dataB.shape)
    # dataB = data[np.where()]
    idxA = np.random.choice(dataA.shape[0], (100 - perc_A_removed) * dataA.shape[0] // 100)
    idxB = np.random.choice(dataB.shape[0], (100 - perc_B_removed) * dataB.shape[0] // 100)
    # idx = np.random.choice(data.shape[0],subset_size*data.shape[0]//100)
    data = np.append(dataA[idxA], dataB[idxB], axis=0)
    np.random.shuffle(data)

    return data