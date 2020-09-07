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

# 3.1.2