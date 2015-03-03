#!/usr/bin/env python
# encoding: utf-8

from data_process import gen_land_pool
from learner import MLLearner
from util import dat_size
from util import learning_curve

import util

from config import N_ITER
from config import ITER_ENABLE

import pickle as pk
import numpy as np
from sklearn.linear_model import LogisticRegression
from config import T
from config import INS_SIZE
from active_learn import model_score
from numpy.random import shuffle

from util import load_test
from util import load_init
from util import load_pool



######## Panda ######
# Comparing Multiple Active Learner vs ELLA + ATS vs ELLA + ATS + AL vs
# ELLA + AL
# This file runs Pure Multiple Active Learner
#####################

###### Load Data ######
pool_dat = load_pool()
init_dat = load_init()
test_dat = load_test()

init_size = dat_size(init_dat)

train_pool = np.array(gen_land_pool(pool_dat))
shuffle(train_pool)

###### Train Initial Model ######

learner = MLLearner(LogisticRegression(), init_dat)


print "Start training..."

print "pool", len(train_pool)
total_pool_size = len(train_pool)
test_acc = []
learned_size= []

init_acc = util.model_roc_score(learner, test_dat)
test_acc = [init_acc]
learned_size = [init_size]



### Training Until No more data available OR Reach the set N_ITER ###
count = N_ITER
while train_pool.size:
    # print "pool", len(train_pool)
    tr_size = min(INS_SIZE, len(train_pool))
    selected = train_pool[:INS_SIZE, :]

    next_train_x = [[] for i in range(0, T)]
    next_train_y = [[] for i in range(0, T)]

    for row in selected:
        t = int(row[10])
        next_train_x[t].append(row[:9])
        next_train_y[t].append(row[9])

    for t in range(0, T):
        if next_train_x[t]:
            learner.refine_model(np.array(next_train_x[t]), np.array(next_train_y[t]), t)

    train_pool = train_pool[INS_SIZE:, :]
    acc = util.model_roc_score(learner, test_dat)


    test_acc.append(acc)
    learned_size.append(total_pool_size - len(train_pool) + init_size)

    if ITER_ENABLE:
        if count < 0: break
        count -= 1

print test_acc
print learned_size
util.learning_curve("res/fig_non_active.png", test_acc, learned_size)
util.curve_to_csv("res/non_ac.csv", test_acc, learned_size)


# for model in models:
    # print "size ", model.get_trained_size()

