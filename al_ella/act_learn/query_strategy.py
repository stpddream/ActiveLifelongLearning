#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from learner import ELLA
from learner import MLLearner
from config import T
import util

def model_uncert(learner, feature, t):
    # return -logis_prob(feature, model)
    return -logis_prob(learner, feature, t)


def logis_prob(learner, feature, t):
    pred = learner.predict_proba(feature, t)

    if isinstance(learner, MLLearner):
        pred = pred[:, 1]

    # For multi label classifier
    # val = np.ndarray.min(np.absolute(np.subtract(pred, 0.5)))
    val = abs(pred - 0.5)
    return val


def dis(x, v):
    """
    Calculate the distance between point and a hyperplane v
    v is the normal vector for the hyperplane, i.e ax + by + c = 0,
    v = [a, b]
    """

    w = np.subtract(0, np.subtract(x, v))
    return np.linalg.norm(np.dot(v, w)) / np.linalg.norm(v)
