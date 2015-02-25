#!/usr/bin/env python
# encoding: utf-8

import numpy as np

class AbstractActiveLearner:
    def __init__(self, base_learner, training_x, training_y):
        self.base_learner = base_learner

    def train_next_set(self):
        raise NotImplementedError("This is abstract class. Not implemented")


class USActiveLearner(AbstractActiveLearner):
    """
    Active Learner utilizes Uncertainty Sampling Method
    """

    def __init__(self, base_learner, training_x, training_y, init_size, step):

        if len(training_x) != len(training_y):
            raise ValueError("X and Y need to be same length")

        self.base_learner = base_learner

        zero_scores = np.zeros(len(training_x))
        self.train_pool = np.column_stack((training_x, training_y, zero_scores))
        print "--- train_pool size ---", len(self.train_pool)

        self.init_size = init_size
        self.train_size = step
        self.model = None
        self.x_train = []
        self.y_train = []

        print "Init complete."

    def has_next(self):
        return len(self.train_pool) > 0

    def train_next_set(self):
        if not self.has_next():
            raise StopIteration

        train_size = 0

        # Randomly Pick in the first iteration
        if not self.model:
            train_size = self.init_size
            train_set_arr = np.array(self.train_pool[:self.init_size])

            self.x_train = train_set_arr[:, :-2]
            self.y_train = train_set_arr[:, -2]
            self.train_pool = self.train_pool[self.init_size:]
            print self.x_train
            print "Model instantiated."

        else:
            train_size = min(self.train_size, len(self.train_pool))
            # Calculate Confidence Level for each data
            coefs = self.model.coef_
            for row in self.train_pool:
                distances = []

                try:
                    for coef in coefs:
                        distances.append(self.dis(row[:-2], np.squeeze(coef)))
                except TypeError:
                    distances.append(self.dis(row[:-2], np.squeeze(coefs)))

                # dis_sorted = sorted(distances)
                # min_dis = dis_sorted[0]
                # sec_min_dis = dis_sorted[1]
                # row[-1] = 1.0 / min_dis - sec_min_dis

                row[-1] = 1.0 / min(distances)

            # Sort by confidence
            sorted_data = self.train_pool[np.argsort(self.train_pool[:, -1])[::-1]]

            # Get the best [train_size] data
            train_set_arr = np.array(sorted_data[:train_size])
            self.train_pool = np.array(sorted_data[train_size:])

            self.x_train = np.vstack((self.x_train, train_set_arr[:, :-2]))
            self.y_train = np.append(self.y_train, train_set_arr[:, -2])

        # Train Model
        self.model = self.base_learner.fit(self.x_train, np.reshape(self.y_train,
            self.y_train.shape[0]))

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



