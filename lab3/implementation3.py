# AUTHOR: boris bubla


# IMPORTS
import numpy as np


# class definition

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
        return attractors.shape[0]

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


# visualise
def visualise_pattern(original_pattern, distorted_pattern, recalled_pattern, image_shape=(32, 32)):
    fig = plt.figure()
    fig.suptitle('Effects of hopfield network', y=0.8)
    ax1 = fig.add_subplot(131)
    ax1.imshow(original_pattern.reshape(image_shape), interpolation='nearest', cmap=cm.Greys_r)
    ax1.title.set_text('Original Pattern')
    ax2 = fig.add_subplot(132)
    ax2.imshow(distorted_pattern.reshape(image_shape), interpolation='nearest', cmap=cm.Greys_r)
    ax2.title.set_text('Degraded')
    ax3 = fig.add_subplot(133)
    ax3.imshow(recalled_pattern.reshape(image_shape), interpolation='nearest', cmap=cm.Greys_r)
    ax3.title.set_text('Reconstructed')


def visualise_random_pattern(original_patterns, distorted_pattern, recalled_pattern, image_shape=(32, 32)):
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle('Effects of hopfield network', y=0.8)
    ax1 = fig.add_subplot(1, original_patterns.shape[0] + 2, 1)
    ax1.imshow(distorted_pattern.reshape(image_shape), interpolation='nearest', cmap=cm.Greys_r)
    ax1.title.set_text('Random')
    ax2 = fig.add_subplot(1, original_patterns.shape[0] + 2, 2)
    ax2.imshow(recalled_pattern.reshape(image_shape), interpolation='nearest', cmap=cm.Greys_r)
    ax2.title.set_text('Reconstruction')
    for i in range(original_patterns.shape[0]):
        ax3 = fig.add_subplot(1, original_patterns.shape[0] + 2, i + 3)
        ax3.imshow(original_patterns[i].reshape(image_shape), interpolation='nearest', cmap=cm.Greys_r)

