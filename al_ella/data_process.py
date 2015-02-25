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

# exit()

# landmine = loadmat('data/LandmineData.mat')

# ### Data Summary ####
# counts = np.zeros(2)
# for t in range(0, T):
    # counts += np.bincount(landmine['label'][:, t][0].reshape(landmine['label'][:, t][0].shape[0]))
# print counts

# print "----> Loading & Splitting data..."
# land_train, land_test = load_landset()
# print "init dat"
# init_dat, pool_dat = gen_init(land_train, 5)

# print "train size", dat_size(land_train)
# print "test  size", dat_size(land_test)
# print "pool  size", dat_size(pool_dat)
# print "init  size", dat_size(init_dat)

# print "----> Writing to files..."

# # Save to files
# pool_f = open("data/pool", "wb")
# pk.dump(pool_dat, pool_f)
# pool_f.close()

# test_f = open("data/test", "wb")
# pk.dump(land_test, test_f)
# test_f.close()

# init_f = open("data/init", "wb")
# pk.dump(init_dat, init_f)
# init_f.close()

# print "----> Landmine Data Preparation Done."

# # # ###### File Load Tests ######
# # # t_pool_f = open("data/pool", "rb")
# # t_pool_dat = pk.load(t_pool_f)
# # t_pool_f.close()
# # print "pool size", dat_size(t_pool_dat)

# # t_init_f = open("data/init", "rb")
# # t_init_dat = pk.load(t_init_f)
# # t_init_f.close()
# # print "init size", dat_size(t_init_dat)

# # t_test_f = open("data/test", "rb")
# t_test_dat = pk.load(t_test_f)
# t_test_f.close()
# print "test size", dat_size(t_test_dat)

