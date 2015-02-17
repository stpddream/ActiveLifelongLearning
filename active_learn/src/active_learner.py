#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import sys


class AbstractActiveLearner:
    def __init__(self, base_learner, training_x, training_y):
        self.base_learner = base_learner

    def train_next_set(self):
        raise NotImplementedError("This is abstract class. Not implemented")


class USActiveLearner(AbstractActiveLearner):
    """
    Active Learner utilizes Uncertainty Sampling Method
    """

    def __init__(self, base_learner, training_x, training_y, init_perc, percent):

        init_size = int(init_perc * 0.01 * len(training_x))
        if init_size > len(training_x):
            raise ValueError("init size larger than pool size")

        if len(training_x) != len(training_y):
            raise ValueError("X and Y need to be same length")

        self.base_learner = base_learner

        zero_scores = np.zeros(len(training_x) - init_size)
        self.train_pool = np.column_stack((training_x[init_size:, ], training_y[init_size:, ], zero_scores))
        print "--- train_pool size ---", len(self.train_pool)

        self.train_size = np.ceil(percent * 0.01 * len(training_x))

        self.x_train = training_x[:init_size]
        self.y_train = training_y[:init_size]
        self.model = self.base_learner.fit(self.x_train, self.y_train)

        print "init complete."

    def has_next(self):
        return len(self.train_pool) > 0

    def train_next_set(self):
        if not self.has_next():
            raise StopIteration

        # Calculate Confidence Level for each data
        coef = self.model.coef_
        for row in self.train_pool:
            row[-1] = self.dis(row[:-2], np.squeeze(coef))

        # Sort by confidence
        sorted_data = self.train_pool[np.argsort(self.train_pool[:, -1])[::-1]]
        train_size = min(self.train_size, len(self.train_pool))

        # Get the best [train_size] data
        train_set_arr = np.array(sorted_data[:train_size])
        self.x_train = np.vstack((self.x_train, train_set_arr[:, :-2]))
        self.y_train = np.append(self.y_train, train_set_arr[:, -2])

        # print "training size:", len(self.x_train)
        # print "pool size:", len(self.train_pool)

        self.model = self.base_learner.fit(self.x_train, self.y_train)

        self.train_pool = np.array(sorted_data[train_size:])
        return train_size

    def get_pool_size(self):
        return len(self.train_pool)

    def get_trained_size(self):
        """
        Get the training size for the last fit
        """
        return len(self.x_train)

    @staticmethod
    def dis(x, v):
        """
        Calculate the distance between point and a hyperplane v
        """

        w = np.subtract(0, np.subtract(x, v))
        return np.linalg.norm(np.dot(v, w)) / np.linalg.norm(v)

    def score(self, test_x, test_y):
        return self.model.score(test_x, test_y)



