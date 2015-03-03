#!/usr/bin/env python
# encoding: utf-8

from data_process import gen_land_pool
from learner import Learner
from config import ITER_ENABLE
from config import N_ITER

import numpy as np
import util
from sklearn.linear_model import LogisticRegression
from config import T
from config import INS_SIZE
from active_learn import comp_info_values
from active_learn import model_uncert
from active_learn import model_score
from numpy.random import shuffle

from util import load_test
from util import load_init
from util import load_pool

# from sklearn import SVM

######## Panda ######
# Comparing Multiple Active Learner vs ELLA + ATS vs ELLA + ATS + AL vs
# ELLA + AL
# This file runs Pure Multiple Active Learner
#####################

###### Load Data ######
pool_dat = load_pool()
init_dat = load_init()
test_dat = load_test()

init_size = util.dat_size(init_dat)

train_pool = gen_land_pool(pool_dat)
shuffle(train_pool)

###### Train Initial Model ######
models = []

for t in range(0, T):
    models.append(Learner(LogisticRegression(), init_dat['feature'][t],
        init_dat['label'][t]))

print "Start training..."
init_acc = model_score(models, test_dat)


print "pool", len(train_pool)
total_pool_size = len(train_pool)
test_acc = [init_acc]
learned_size = [init_size]

count = N_ITER
### Training Until No more data available ###
while train_pool.size:
    # print "pool", len(train_pool)
    tr_size = min(INS_SIZE, len(train_pool))
    train_pool = comp_info_values(models, train_pool, model_uncert)
    sorted_dat = train_pool[np.argsort(train_pool[:, -1])[::-1]]
    selected = sorted_dat[:INS_SIZE, :]

    next_train_x = [[] for i in range(0, T)]
    next_train_y = [[] for i in range(0, T)]

    for row in selected:
        t = int(row[10])
        next_train_x[t].append(row[:9])
        next_train_y[t].append(row[9])

    for t in range(0, T):
        if next_train_x[t]:
            models[t].refine_model(np.array(next_train_x[t]), np.array(next_train_y[t]))

    train_pool = sorted_dat[INS_SIZE:, :]
    acc = model_score(models, test_dat)
    test_acc.append(acc)
    learned_size.append(total_pool_size - len(train_pool) + init_size)

    # print acc
    # print models_act(models)

    if ITER_ENABLE:
        if count < 0: break
        count -= 1

print test_acc
print learned_size
util.learning_curve("res/active_mult_fig.png", test_acc, learned_size)
util.curve_to_csv("res/ac_multi.csv", test_acc, learned_size)



# for model in models:
    # print "size ", model.get_trained_size()

