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

import copy
import config

######## Panda ######
# Comparing Multiple Active Learner vs ELLA + ATS vs ELLA + ATS + AL vs
# ELLA + AL
# This file runs Pure Multiple Active Learner
#####################

### Task Control Constants ###
TASK_NONE = 0
TASK_RAND = -1
TASK_ACTIVE = 1
##############################

def model_eval(learner, test_dat, eval_method):
    if eval_method == config.EVAL_ACC:
        acc = ln.model_score(learner, test_dat)
    else:
        acc = ln.model_roc_score(learner, test_dat)
    return acc


def run_lm_trials(pool_dat, init_dat, test_dat, learner_class, engine=None, do_active=False,
        task_ctrl=TASK_NONE, evaluation=config.EVAL_ACC, n_trials=1):
    scores = []
    for i in range(0, n_trials):
        print 'trial', i
        score, learned_size = run_ml_landmine(pool_dat, init_dat, test_dat, learner_class, evaluation=evaluation,
                engine=engine, do_active=do_active, task_ctrl=task_ctrl)
        scores.append(score)
    return np.mean(np.array(scores), axis=0), learned_size




def run_ml_landmine(pool_dat, init_dat, test_dat, learner_class, engine=None, do_active=False, task_ctrl=TASK_NONE, evaluation=config.EVAL_ACC):

    if task_ctrl == TASK_NONE:
        active_task = False
        rand_task = False
    elif task_ctrl == TASK_RAND:
        active_task = False
        rand_task = True
    elif task_ctrl == TASK_ACTIVE:
        active_task = True
        rand_task = False


    # Copy all data
    test_dat_cp = copy.deepcopy(test_dat)
    init_dat_cp = copy.deepcopy(init_dat)
    pool_dat_cp = copy.deepcopy(pool_dat)


    ###### Train Initial Model ######
    init_size = dat_size(init_dat_cp)

    if learner_class is ELLA:
        # Add bias term
        test_dat_cp = util.add_bias(test_dat_cp)
        pool_dat_cp = util.add_bias(pool_dat_cp)
        init_dat_cp = util.add_bias(init_dat_cp)

        learner = learner_class(engine, init_dat_cp)
    else: learner = learner_class(LogisticRegression(), init_dat_cp)

    print "Start training..."
    test_acc = []
    learned_size= []

    prof = Professor(init_dat_cp, pool_dat_cp, random=True, do_active=do_active, multi_t=active_task or rand_task, rand_task=rand_task)
    total_pool_size = prof.get_pool_size()

    init_acc = model_eval(learner, test_dat_cp, evaluation)
    test_acc = [init_acc]
    learned_size = [init_size]
    task_rec = []

    ## Training Until No more data available OR Reach the set N_ITER ###
    count = N_ITER
    while prof.has_next():
        # print "pool", prof.get_pool_size()
        selected_x, selected_y, tasks = prof.next_train_set(INS_SIZE, learner=learner)
        task_rec.append(tasks[0])

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

        acc = model_eval(learner, test_dat_cp, evaluation)
        # print acc
        test_acc.append(acc)
        # print learner.get_trained_size()
        learned_size.append(total_pool_size - prof.get_pool_size() + init_size)

        if ITER_ENABLE:
            if count < 0: break
            count -= 1

        if count % 20 == 0:
            if engine is not None:
                engine.clean_mem()

    util.curve_to_csv("res/task_save" + str(do_active) + " .csv", task_rec, [0])

    return test_acc, learned_size
