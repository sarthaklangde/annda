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

    # keep making datasets until nonlin sep dataset is made ; will enter infinite loop if you choose shitty mean and sigmas
    separable = True
    while separable:
        # create data
        A1 = np.array(
            [np.random.normal(-mA[0], sigmaA, (n // 4, 1)), np.random.normal(mA[1], sigmaA, (n // 4, 1))]).reshape(
            n // 4, 2)
        A2 = np.array(
            [np.random.normal(mA[0], sigmaA, (n // 4, 1)), np.random.normal(mA[1], sigmaA, (n // 4, 1))]).reshape(
            n // 4, 2)
        A = np.append(A1, A2, axis=0)
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
