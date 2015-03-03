#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from config import T
import copy


class AbstractLearner():

    def __init__(self, base_learner, init_dat):
        pass

    def refine_model(self, feature, target):
        pass

    def score(self, test_x, test_y):
        pass

    def get_pool_size(self):
        pass

    def get_trained_size(self):
        pass


class MLLearner(AbstractLearner):
    """
    Multi Task Learner
    """

    def __init__(self, base_learner, init_dat):

        self.models = [copy.copy(base_learner) for t in range(0, T) ]

        self.x_train = []
        self.y_train = []

        # Init
        for t in range(0, T):
            self.x_train.append(init_dat['feature'][t])
            self.y_train.append(init_dat['label'][t])
            self.models[t].fit(self.x_train[t], self.y_train[t])


    def refine_model(self, Xs, Ys, t):
        self.x_train[t] = np.vstack((self.x_train[t], Xs))
        self.y_train[t] = np.concatenate((self.y_train[t], Ys))

        self.models[t] = self.models[t].fit(self.x_train[t], self.y_train[t])


    def predict_proba(self, x_test, t):
        return self.models[t].predict_proba(x_test)


    def score(self, x_test, y_test, t):
        return self.models[t].score(x_test, y_test)


    def get_pool_size(self):
        return len(self.train_pool)

    def get_trained_size(self):
        """
        Get the training size for the last fit
        """
        return len(self.train_x)

    def get_model(self, t):
        return self.models[t]

    def get_model_coef(self):
        return np.reshape(self.model.coef_, self.model.coef_.shape[1])

    def is_activated(self):
        return self.activated

    @staticmethod
    def dis(x, v):
        """
        Calculate the distance between point and a hyperplane v
        """

        w = np.subtract(0, np.subtract(x, v))
        return np.linalg.norm(np.dot(v, w)) / np.linalg.norm(v)


