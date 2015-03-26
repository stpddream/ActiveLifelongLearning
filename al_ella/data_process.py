#!/usr/bin/env python
# encoding: utf-8

from scipy.io import loadmat
from sklearn.cross_validation import train_test_split
import numpy as np
import pickle as pk
from config import T
from config import TRAIN_PERC
from util import dat_size
import util


def load_dat(filename):
    mat = loadmat('data/' + filename)
    return split_dat(mat, TRAIN_PERC, mat=True)


def split_dat(dat, perc, mat=False):

    set_one = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}
    set_two = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}

    for t in range(0, T):
        # print "t", t
        if mat:
            task_feat = dat['feature'][:,t][0]
            task_target = dat['label'][:,t][0]
        else:
            task_feat = dat['feature'][t]
            task_target = dat['label'][t]

        while True:
            x_one, x_two, y_one, y_two = train_test_split(task_feat, task_target, train_size=perc)
            if util.contain_one(y_one.reshape(y_one.shape[0])) and util.contain_one(y_two.reshape(y_two.shape[0])):
                break
            # print "Retry"

        set_one['feature'][t] = x_one
        set_one['label'][t] = np.reshape(y_one, y_one.shape[0])
        set_two['feature'][t] = x_two
        set_two['label'][t] = np.reshape(y_two, y_two.shape[0])
        # print y_one

    return set_one, set_two



def load_landset():
    landmine = loadmat('data/LandmineData.mat')
    return split_dat(landmine, TRAIN_PERC, mat=True)


# Return init_set, pool_set
def gen_init(train_dat, init_size):
    return split_dat(train_dat, init_size)

def gen_land_pool(train_dat, multi_t=False):
    dat_ret = []

    for t in range(0, T):
        task_feat = train_dat['feature'][t]
        task_target = train_dat['label'][t]
        cols = train_dat['feature'][t].shape[0]

        task_label = np.array([t for i in range(cols)])
        ins_value = np.zeros(cols)
        task_dat = np.column_stack((np.column_stack((np.column_stack((task_feat, task_target)), task_label)),
                ins_value))

        if multi_t:
            dat_ret.append(task_dat)
        else: dat_ret.extend(task_dat)

    return np.array(dat_ret)


def lab_count(dat):
    count = 0
    for t in range(0, T):
        count += np.bincount(dat['label'][:, t][0].reshape(dat['label'][:, t][0].shape[0]))
    print count
