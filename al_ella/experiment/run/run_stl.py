#!/usr/bin/env python
# encoding: utf-8

from data_process import gen_land_pool
from learner import MLLearner
from util import dat_size
from util import learning_curve

import matplotlib.pyplot as plt

import util
from act_learn import Professor

from config import N_ITER
from config import ITER_ENABLE
from config import T
from config import INS_SIZE
import config



import numpy as np
from sklearn.linear_model import LogisticRegression
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

def run_stl_landm(pool_dat, init_dat, test_dat, do_active):

###### Train Initial Model ######
    init_size = dat_size(init_dat)
    learner = MLLearner(LogisticRegression(), init_dat)

    print "Start training..."

    print "pool", len(train_pool)
    total_pool_size = len(train_pool)
    test_acc = []
    learned_size= []

    prof = Professor(init_dat, pool_dat, random=True, do_active=do_active)
    total_pool_size = prof.get_pool_size()

    init_acc = util.model_roc_score(learner, test_dat)
    test_acc = [init_acc]
    learned_size = [init_size]

### Training Until No more data available OR Reach the set N_ITER ###
    count = N_ITER
    while prof.has_next():
        # print "pool", len(train_pool)
        selected_x, selected_y, tasks = prof.next_train_set(INS_SIZE, learner=learner)

        # Group selected training set by tasks
        next_train_x = [[] for i in range(0, T)]
        next_train_y = [[] for i in range(0, T)]

        for i in range(0, selected_x.shape[0]):
            t = int(tasks[i])
            next_train_x[t].append(selected_x[i, :])
            next_train_y[t].append(selected_y[i])


        for t in range(0, T):
            if next_train_x[t]:
                learner.refine_model(np.array(next_train_x[t]), np.array(next_train_y[t]), t)

        # acc = util.model_roc_score(learner, test_dat)
        acc = util.model_score(learner, test_dat)

        test_acc.append(acc)
        learned_size.append(total_pool_size - prof.get_pool_size() + init_size)

        if ITER_ENABLE:
            if count < 0: break
            count -= 1


    return test_acc, learned_size


pool_dat = load_pool()
init_dat = load_init()
test_dat = load_test()

train_pool = np.array(gen_land_pool(pool_dat))
shuffle(train_pool)

from experiment import run_ml_landmine
print "[info]Start passive learning..."
test_acc_ps, learned_size_ps = run_ml_landmine(pool_dat, init_dat, test_dat, MLLearner, do_active=False)
util.curve_to_csv("res/ps_stl_non.csv", test_acc_ps, learned_size_ps)


# print "[info]Start passive learning..."
# test_acc_ps, learned_size_ps = run_stl_landm(pool_dat, init_dat, test_dat, do_active=False)
# util.curve_to_csv("res/ps_stl_non.csv", test_acc_ps, learned_size_ps)

# print "[info]Start active learning..."
# test_acc_ac, learned_size_ac = run_stl_landm(pool_dat, init_dat, test_dat, do_active=True)
# util.curve_to_csv("res/acc_stl_non.csv", test_acc_ac, learned_size_ac)

# print test_acc_ps
# print learned_size_ps

# print test_acc_ac
# print learned_size_ac

figure = plt.figure()
line_stl_ps, = plt.plot(learned_size_ps, test_acc_ps, label='Non Active')
line_stl_ac, = plt.plot(learned_size_ac, test_acc_ac, label='Active')
plt.legend(loc='lower right', handles=[line_stl_ps, line_stl_ac])
plt.xlabel('Size of training set')
plt.ylabel('Accuracy')
plt.savefig('res/uncert_log_prob/' + 'acc_' + str(config.TRAIN_PERC) + '_' + str(config.INS_SIZE) + '_stl.png')

