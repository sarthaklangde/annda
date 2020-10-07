import os
import copy
import numpy as np
import matplotlib.pyplot as plt


class hopfield_network():
    """
    A class for applying hopfield networks, via synchronous or sequential updates.
    Inputs:
    patterns = the patterns (nxd) you wish to memorise
    update_rule = 'synchronous' or 'sequential'
    Methods:
    recall - reconstructs a pattern from a given distorted pattern
    attractor_search - random sampling based search for attractors
    attractor_search2 - search for attractors by systematically trying each possible distorted pattern
    Returns:
    hopfield_network object
    """

    def __init__(self, patterns, update_rule='synchronous'):

        self.patterns = patterns
        self.weights = np.dot(patterns.T, patterns)
        # print(self.weights.shape)
        self.update_rule = update_rule
        self.check_capacity()

    def check_capacity(self):
        # checks whether input patterns are stable / fixed points

        # x = self.update(patterns=self.patterns)
        x = np.dot(self.weights, self.patterns.T)
        x = (x >= 0) * 1
        x = np.where(x == 0, -1, x)
        assert (
                    x.T == self.patterns).all(), """The network cannot store all these patterns! Ie patterns are not stable/fixed points!"""

    def update(self, patterns):
        # learning rule

        if self.update_rule == 'synchronous':
            x = np.dot(self.weights, patterns.T)
            x = (x >= 0) * 1
            x = np.where(x == 0, -1, x)
            # print(x)
            return x.T

        if self.update_rule == 'sequential':

            y = copy.deepcopy(patterns)

            for cnt, i in enumerate(self.weights):
                x_i = np.dot(i, y.T)
                x_i = (x_i >= 0) * 1
                x_i = np.where(x_i == 0, -1, x_i)
                # print(x_i)
                y[cnt] = x_i

            return y

    def recall(self, distorted_patterns, max_iter):
        # 1. tries to find stored patterns from distorted patterns up to max_iter times
        # 2. stops if fixed point is reached or pattern is correctly recalled

        x = copy.deepcopy(distorted_patterns)

        for i in range(max_iter):

            x_new = self.update(x)

            #             if (x_new==self.patterns).all():
            #                 print('Patterns recalled after {} iterations'.format(i+1))
            #                 #print('Distorted Patterns:\n',x_new)
            #                 #print('Stored Patterns:\n',self.patterns)
            #                 break

            if (x_new == x).all():
                print('Fixed point reached after {} iterations'.format(i + 1))
                # print('Distorted Patterns:\n',x_new)
                # print('Stored Patterns:\n',self.patterns)
                break

            x = x_new

        # returns the last found pattern - either a fixed point or an original pattern
        # compare it to the original patterns to which it is
        return x_new

    def attractor_search(self, max_bit_error=3, search_iter=100):
        # 1. changes stored patterns randomly by up to max_bit_error bits
        # 2. then tries to converge to the original stored pattern again
        # 3. keeps count of how many different convergence points there are (ie num of attractors)
        # 4. returns number of attractors

        convergence_points = []

        for j in range(search_iter):
            base_pattern_idx = np.random.choice(
                np.arange(self.patterns.shape[0]))  # chooses one of n stored patterns to distort
            x = self.patterns[base_pattern_idx]  # x var is our soon to be distorted pattern

            # distort pattern
            for i in range(max_bit_error):
                flip = np.random.choice([-1, 1])
                loc = np.random.choice(np.arange(self.weights.shape[1]))
                x[loc] = flip

            # try to find original pattern - maybe change max_iter here if more complex patterns and datasets
            x_new = self.recall(x, max_iter=50)
            convergence_points.append(x_new)

        # print('CONVERGENCE POINTS:\n',np.vstack(convergence_points))
        attractors = np.unique(np.vstack(convergence_points), axis=0)

        print('These are the attractors:\n', attractors)

        # return num of attractors
        return attractors

    def attractor_search2(self):
        # 1. create all possible distorted signals - d^2
        # 2. then try to recall a pattern from each
        # 3. count of how many different convergence points there are (ie num of attractors)

        convergence_points = []

        # 1.
        d = np.arange(2 ** (self.weights.shape[0]))
        m = self.weights.shape[0]
        all_signals = (((d[:, None] & (1 << np.arange(m)))) > 0).astype(int)  # all signals as binary
        print(all_signals.shape)
        all_signals = np.where(all_signals == 0, -1, all_signals)

        for i in range(2 ** (self.weights.shape[0])):
            # 2.
            x_new = self.recall(all_signals[i], max_iter=50)
            convergence_points.append(x_new)
        # 3.
        attractors = np.unique(np.vstack(convergence_points), axis=0)
        print('These are the attractors:\n', attractors)

        # return num of attractors
        return attractors.shape[0]


def flipbits(pattern, percentage=0.1):
    tot = len(pattern)
    new_pattern = copy.deepcopy(pattern)
    num = int(percentage * tot)
    idx = np.arange(tot)
    idx_to_flip = np.random.choice(idx, num, replace=False)
    for i in range(num):
        idd = idx_to_flip[i]
        if new_pattern[idd] == 1.:
            new_pattern[idd] = -1.
        else:
            new_pattern[idd] = 1.
    return new_pattern


def noise_exp_batch(pattern, task):
    percentages = np.arange(0.3, 0.51, 0.01)
    for i in range(len(percentages)):
        pattern_noised = flipbits(pattern, percentages[i])
        pattern_restructured = task.recall(pattern_noised, max_iter=1)
        visualize_pattern(pattern, pattern_noised, pattern_restructured, percentages[i])


def main():
    cwd = os.getcwd()
    path_to_data = cwd + '/pict.dat'
    pictures = np.genfromtxt(path_to_data, delimiter=',')
    pictures = pictures.reshape(11, 1024)
    p1 = pictures[0]
    p2 = pictures[1]
    p3 = pictures[2]
    image_shape = (32, 32)


    task = hopfield_network(pictures[0:3])

    # noise_exp_batch(p1, task)
    attractors = task.attractor_search(max_bit_error=1024)
    num = attractors.shape[0]
    print("Num", num)
    fig = plt.figure()
    ax = []
    for i in range(num):
        ax.append(fig.add_subplot(1, num, i+1))
        ax[i].imshow(attractors[i].reshape(image_shape), interpolation='nearest', cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()

def visualize_pattern(p1, p2, p3, percentage, image_shape=(32, 32)):
    fig = plt.figure(figsize=(plt.figaspect(0.3)))
    fig.suptitle('{}% noise in pattern'.format(int(percentage * 100)), y=1)
    ax1 = fig.add_subplot(131)
    ax1.imshow(p1.reshape(image_shape), interpolation='nearest', cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.title.set_text('Original Pattern')

    ax2 = fig.add_subplot(132)
    ax2.imshow(p2.reshape(image_shape), interpolation='nearest', cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.title.set_text('Degraded')

    ax3 = fig.add_subplot(133)
    ax3.imshow(p3.reshape(image_shape), interpolation='nearest', cmap='gray')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.title.set_text('Reconstructed')
    plt.tight_layout()
    plt.show()


main()
