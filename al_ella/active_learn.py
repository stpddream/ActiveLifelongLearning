#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from config import T

def comp_info_values(models, features, func):
    """
    Compute information value based on coef
    """
    return np.apply_along_axis(func, 1, features, models)

def model_uncert(feature, models):
    # print feature
    t = feature[10] # Get the task # for the feature
    # print "comp between task", t
    feature[11] = dis(feature[:9], models[int(t)].get_model_coef())
    return feature


def dis(x, v):
    """
    Calculate the distance between point and a hyperplane v
    """

    w = np.subtract(0, np.subtract(x, v))
    return np.linalg.norm(np.dot(v, w)) / np.linalg.norm(v)

def model_score(models, tests):
    total_score = 0.0

    for t in range(0, T):
        this_score = models[t].score(tests['feature'][t], tests['label'][t])
        # print t, ":", this_score
        total_score += this_score

    return total_score / T





