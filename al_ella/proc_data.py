#!/usr/bin/env python
# encoding: utf-8

from scipy.io import loadmat
from sklearn.cross_validation import train_test_split
import numpy as np
import pickle as pk
from config import T
from config import TRAIN_PERC
from util import dat_size

def split_dat(dat, perc, mat=False):

    set_one = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}
    set_two = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}

    for t in range(0, T):
        if mat:
            task_feat = dat['feature'][:,t][0]
            task_target = dat['label'][:,t][0]
        else:
            task_feat = dat['feature'][t]
            task_target = dat['label'][t]

        x_one, x_two, y_one, y_two = train_test_split(task_feat, task_target, train_size=perc)

        set_one['feature'][t] = x_one
        set_one['label'][t] = np.reshape(y_one, y_one.shape[0])
        set_two['feature'][t] = x_two
        set_two['label'][t] = np.reshape(y_two, y_two.shape[0])

    return set_one, set_two


def load_landset():
    landmine = loadmat('data/LandmineData.mat')
    return split_dat(landmine, TRAIN_PERC, mat=True)


# def load_landmine():
    # """
     # Load land mine data into memory and split into train and test sets
    # """

    # landmine = loadmat('data/LandmineData.mat')

    # train_set = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}
    # test_set = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}

    # t_n_tests = 0
    # t_n_train = 0

    # for t in range(0, T):
        # task_feat = landmine['feature'][:,t][0]
        # task_target = landmine['label'][:,t][0]

        # x_train, x_test, y_train, y_test = train_test_split(task_feat, task_target, train_size=TRAIN_PERC)

        # train_set['feature'][t] = x_train
        # train_set['label'][t] = np.reshape(y_train, y_train.shape[0])
        # test_set['feature'][t] = x_test
        # test_set['label'][t] = np.reshape(y_test, y_test.shape[0])

        # t_n_tests += x_test.shape[0]
        # t_n_train += x_train.shape[0]

    # print "train size", t_n_train
    # print "test  size", t_n_tests

    # return train_set, test_set

def prod_init_dat(train_dat, init_size):
    train_pl = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}
    init_dat = {'feature':[[] for i in range(0, T)], 'label':[[] for i in range(0, T)]}

    t_n_init = 0
    t_n_pool = 0

    for t in range(0, T):
        task_feat = train_dat['feature'][t]
        task_target = train_dat['label'][t]
        train_pl_x, init_x, train_pl_y, init_y = train_test_split(task_feat, task_target,
                test_size=init_size)

        train_pl['feature'][t] = train_pl_x
        train_pl['label'][t] = np.reshape(train_pl_y, train_pl_y.shape[0])
        init_dat['feature'][t] = init_x
        init_dat['label'][t] = np.reshape(init_y, init_y.shape[0])

        t_n_pool += train_pl_x.shape[0]
        t_n_init += init_x.shape[0]

    print "init  size", t_n_init
    print "pool  size", t_n_pool

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


land_train, land_test = load_landset()
pool_dat, init_dat = prod_init_dat(land_train, 100)

print "train size", dat_size(land_train)
print "test  size", dat_size(land_test)


# Save to files
pool_f = open("data/pool", "wb")
pk.dump(pool_dat, pool_f)
pool_f.close()

test_f = open("data/test", "wb")
pk.dump(land_test, test_f)
test_f.close()

init_f = open("data/init", "wb")
pk.dump(init_dat, init_f)
init_f.close()

print "----> Landmine Data Preparation Done."

###### File Load Tests ######
t_pool_f = open("data/pool", "rb")
t_pool_dat = pk.load(t_pool_f)
print "pool size", dat_size(t_pool_dat)

t_init_f = open("data/init", "rb")
t_init_dat = pk.load(t_init_f)
print "init size", dat_size(t_init_dat)

t_test_f = open("data/test", "rb")
t_test_dat = pk.load(t_test_f)
print "test size", dat_size(t_test_dat)

