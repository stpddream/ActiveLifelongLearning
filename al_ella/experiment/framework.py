#!/usr/bin/env python
# encoding: utf-8

from util import dat_size
import util
from act_learn import Professor

from config import N_ITER
from config import ITER_ENABLE
from config import T
from config import INS_SIZE
import numpy as np
from sklearn.linear_model import LogisticRegression

import learner as ln
from learner import ELLA



######## Panda ######
# Comparing Multiple Active Learner vs ELLA + ATS vs ELLA + ATS + AL vs
# ELLA + AL
# This file runs Pure Multiple Active Learner
#####################

EVAL_ACC = 0
EVAL_AUC = 1

## Evaluation: [0 for Accuracy, 1 for AUC]
def run_ml_landmine(pool_dat, init_dat, test_dat, learner_class, engine=None, do_active=False, rand_task=False, evaluation=EVAL_ACC):

###### Train Initial Model ######
    init_size = dat_size(init_dat)

    if learner_class is ELLA:
        learner = learner_class(engine, init_dat)
    else: learner = learner_class(LogisticRegression(), init_dat)

    print "Start training..."
    test_acc = []
    learned_size= []

    prof = Professor(init_dat, pool_dat, random=True, do_active=do_active, rand_task=rand_task)
    total_pool_size = prof.get_pool_size()

    init_acc = ln.model_roc_score(learner, test_dat)
    test_acc = [init_acc]
    learned_size = [init_size]

### Training Until No more data available OR Reach the set N_ITER ###
    count = N_ITER
    while prof.has_next():
        print "pool", prof.get_pool_size()
        selected_x, selected_y, tasks = prof.next_train_set(INS_SIZE, learner=learner)

        # Group selected training set by tasks
        next_train_x = [[] for i in range(0, T)]
        next_train_y = [[] for i in range(0, T)]

        for i in range(0, selected_x.shape[0]):
            t = int(tasks[i])
            next_train_x[t].append(selected_x[i, :])
            next_train_y[t].append(selected_y[i])

        print "start training..."

        for t in range(0, T):
            if next_train_x[t]:
                learner.refine_model(np.array(next_train_x[t]), np.array(next_train_y[t]), t)

        if evaluation == EVAL_ACC:
            acc = ln.model_score(learner, test_dat)
        else:
            acc = ln.model_roc_score(learner, test_dat)

        test_acc.append(acc)
        learned_size.append(total_pool_size - prof.get_pool_size() + init_size)

        if ITER_ENABLE:
            if count < 0: break
            count -= 1


    return test_acc, learned_size
