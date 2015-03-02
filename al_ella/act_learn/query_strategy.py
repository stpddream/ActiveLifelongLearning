#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from config import T

def model_uncert(feature, model):
    # return -logis_prob(feature, model)
    return -logis_prob(feature, model)


def logis_prob(feature, model):
    val = np.ndarray.min(np.absolute(np.subtract(model.predict_proba(feature), 0.5)))
    return val


def dis(x, v):
    """
    Calculate the distance between point and a hyperplane v
    v is the normal vector for the hyperplane, i.e ax + by + c = 0,
    v = [a, b]
    """

    w = np.subtract(0, np.subtract(x, v))
    return np.linalg.norm(np.dot(v, w)) / np.linalg.norm(v)
