#!/usr/bin/env python
# encoding: utf-8

from scipy.io import loadmat
from sklearn.cross_validation import train_test_split
import numpy as np
from config import T


TRAIN_PERC = 0.8


def load_landmine():
    """
     Load land mine data into memory and split into train and test sets
    """

    landmine = loadmat('data/LandmineData.mat')

    train_set = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}
    test_set = {'feature':[], 'label':[]}

    for t in range(0, T):
        task_feat = landmine['feature'][:,t][0]
        task_target = landmine['label'][:,t][0]

        x_train, x_test, y_train, y_test = train_test_split(task_feat, task_target, train_size=TRAIN_PERC)

        train_set['feature'][t] = x_train
        train_set['label'][t] = np.reshape(y_train, y_train.shape[0])
        test_set['feature'].extend(x_test)
        test_set['label'].extend(np.reshape(y_test, y_test.shape[0]))

    return train_set, test_set

def prod_init_dat(train_dat, init_size):
    train_pl = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}
    init_dat = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}

    for t in range(0, T):
        task_feat = train_dat['feature'][t]
        task_target = train_dat['label'][t]
        train_pl_x, init_x, train_pl_y, init_y = train_test_split(task_feat, task_target,
                test_size=init_size)

        train_pl['feature'][t] = train_pl_x
        train_pl['label'][t] = np.reshape(train_pl_y, train_pl_y.shape[0])
        init_dat['feature'][t] = init_x
        init_dat['label'][t] = np.reshape(init_y, init_y.shape[0])

    return train_pl, init_dat


def gen_land_pool(train_dat):
    dat_ret = []
    for t in range(0, T):
        task_feat = train_dat['feature'][t]
        task_target = train_dat['label'][t]
        cols = train_dat['feature'][t].shape[0]

        task_label = np.array([t for i in range(cols)])
        ins_value = np.zeros(cols)
        task_dat = np.column_stack((np.column_stack((np.column_stack((task_feat, task_target)), task_label)),
                ins_value))

        dat_ret.extend(task_dat)
    return dat_ret


# #### Load land mine test
# train, test = load_landmine()
# print train['feature'][1].shape
# print train['label'][1].shape

# print len(test['feature'])
# print len(test['label'])


# # #### Gen Land Pool test ####
# # train, test = load_landmine()
# gen_land_pool(train)
