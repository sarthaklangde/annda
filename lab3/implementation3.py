# AUTHOR: boris bubla


# IMPORTS
import numpy as np


class hopfield_network():

    def __init__(self, patterns):

        self.patterns = patterns
        self.weights = np.dot(patterns.T, patterns)
        # print(self.weights.shape)
        self.check_capacity()

    def check_capacity(self):
        # checks whether input patterns are stable / fixed points

        x = self.update(patterns=self.patterns)
        assert (
                    x == self.patterns).all(), """The network cannot store all these patterns! Ie patterns are not stable/fixed points!"""

    def update(self, patterns):
        # learning rule

        x = np.dot(self.weights, patterns.T)
        x = (x >= 0) * 1
        x = np.where(x == 0, -1, x)
        # print(x)
        return x.T

    def recall(self, distorted_patterns, max_iter):
        # 1. tries to find stored patterns from distorted patterns up to max_iter times
        # 2. stops if fixed point is reached or pattern is correctly recalled

        x = distorted_patterns

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
