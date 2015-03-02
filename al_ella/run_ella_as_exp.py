#!/usr/bin/env python
# encoding: utf-8

from util import load_test
from util import load_pool
from util import load_init
import util
import numpy as np
import matplotlib.pyplot as plt
from learner import ELLA
from learner import ella_score
from learner import ella_auc_score

from config import T
from config import INS_SIZE
from config import N_ITER
from config import ITER_ENABLE

from data_process import gen_land_pool

import matlab.engine
from act_learn import Professor

print "Starting matlab..."
eng = matlab.engine.start_matlab()

ELLA_DIR = "/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/lib/ELLAv1.0"
eng.addpath("/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/ml")
eng.addpath(eng.genpath(ELLA_DIR))
# res = eng.runExperimentActiveTask()
# print res


######## Panda ######
# Comparing Multiple Active Learner vs ELLA + ATS vs ELLA + ATS + AL vs
# ELLA + AL
# This file runs ELLA + Active Task Selection
#####################

## Load all files
test_dat = util.add_bias(load_test())
pool_dat = util.add_bias(load_pool())
init_dat = util.add_bias(load_init())
init_size = util.dat_size(init_dat)

## Init ELLA Model with init set ##
ella_model = ELLA(eng, init_dat)
init_acc = ella_score(ella_model, test_dat)
test_acc = [init_acc]
learned_size = [init_size]


prof = Professor(init_dat, pool_dat, multi_t=True, random=True)
total_pool_size = prof.get_pool_size()
print "train pool size", total_pool_size

# ### Training Until No more data available OR Reach the set N_ITER ###
count = N_ITER
while prof.has_next():
    # print "pool", prof.get_pool_size()

    # Get next training instances
    selected_x, selected_y, tasks = prof.next_train_set(INS_SIZE, learner=ella_model)

    # Group selected training set by tasks
    next_train_x = [[] for i in range(0, T)]
    next_train_y = [[] for i in range(0, T)]

    for i in range(0, selected_x.shape[0]):
        t = int(tasks[i])
        next_train_x[t].append(selected_x[i, :])
        next_train_y[t].append(selected_y[i])

    # Training #
    for t in range(0, T):
        if next_train_x[t]:
            ella_model.refine_model(np.array(next_train_x[t]), np.array(next_train_y[t]), t)

    # Yield results
    # acc = ella_score(ella_model, test_dat)
    acc = ella_auc_score(ella_model, test_dat)


    test_acc.append(acc)
    learned_size.append(total_pool_size - prof.get_pool_size() + init_size)

    if ITER_ENABLE:
        if count < 0 : break
        count -= 1

print test_acc
print learned_size

util.learning_curve("res/fig_ella_non.png", test_acc, learned_size)
util.curve_to_csv("res/ella_non.csv", test_acc, learned_size)
