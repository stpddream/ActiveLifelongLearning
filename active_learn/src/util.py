#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import sklearn
from sklearn.datasets import *
from scipy.io import loadmat

def shuffle_ins(Xs, Ys):
    shuffled_x = np.empty(Xs.shape)
    shuffled_y = np.empty(Ys.shape)

    indices = np.random.permutation(Xs.shape[0])

    for o_idx, n_idx in enumerate(indices):
        # print shuffled_x[n_idx, :]
        # print Xs[o_idx, :]
        shuffled_x[n_idx, :] = Xs[o_idx, :]
        shuffled_y[n_idx] = Ys[o_idx]

    return shuffled_x, shuffled_y


def read_txt_dat(file_name):
    file = open(file_name, 'rb')
    dat_lines = file.readlines()

    dat = []
    for line in dat_lines:
        dat.append(int(line))
    return dat

def load_mat(file_name, label_name):
    dat = loadmat(file_name)
    return_val = sklearn.datasets.base.Bunch()
    return_val.data = dat['data']

    labels = read_txt_dat(label_name)

    return_val.target = np.array(labels)
    return return_val
