#!/usr/bin/env python
# encoding: utf-8

# from experiment import run_ml_landmine
from experiment import run_lm_trials
from learner import MLLearner
from learner import ELLA
from util import dat_size
from util import learning_curve
import numpy as np
import matlab.engine
import gc

import matplotlib.pyplot as plt

import util
import config
import experiment as exp

from util import load_test
from util import load_init
from util import load_pool

def reload_dat():
    gc.collect()
    pool_dat = load_pool()
    init_dat = load_init()
    test_dat = load_test()
    return pool_dat, init_dat, test_dat

pool_dat, init_dat, test_dat = reload_dat()


#############################
###### STL Experiments ######
#############################

# print "[info]Start passive learning..."
# test_acc_ps, learned_size_ps = run_ml_landmine(pool_dat, init_dat, test_dat, MLLearner, evaluation=config.EVAL_ME, do_active=False)
# util.curve_to_csv("res/ps_stl_non.csv", test_acc_ps, learned_size_ps)

# print "[info]Start active learning..."
# test_acc_ac, learned_size_ac = run_ml_landmine(pool_dat, init_dat, test_dat, MLLearner, evaluation=config.EVAL_ME, do_active=True)
# util.curve_to_csv("res/acc_stl_non.csv", test_acc_ac, learned_size_ac)

# figure = plt.figure()
# line_stl_ps, = plt.plot(learned_size_ps, test_acc_ps, label='Non Active')
# line_stl_ac, = plt.plot(learned_size_ac, test_acc_ac, label='Active')
# plt.legend(loc='lower right', handles=[line_stl_ps, line_stl_ac])
# plt.xlabel('Size of training set')
# plt.ylabel('Accuracy')
# plt.savefig('res/uncert_log_prob/' + 'acc_' + str(config.TRAIN_PERC) + '_' + str(config.INS_SIZE) + '_stl.png')

##############################
###### ELLA Experiments ######
##############################

# Set up environment
print "Starting matlab..."
eng = matlab.engine.start_matlab()

ELLA_DIR = "/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/lib/ELLAv1.0"
eng.addpath("/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/ml")
eng.addpath(eng.genpath(ELLA_DIR))

# Run experiment

figure = plt.figure()
handles = []

fig_cap = ['ellaps', 'ellart', 'ellaat', 'ellaal', 'ellaact']
fig_flag = [False for i in range(0, len(fig_cap))]

# ### Passive ELLA ###
# fig_flag[0] = True

# test_acc_el_ps, learned_size_el_ps = run_lm_trials(pool_dat, init_dat, test_dat, ELLA,
        # evaluation=config.EVAL_ME, engine=eng, do_active=False, task_ctrl=exp.TASK_NONE, n_trials=5)
# print test_acc_el_ps
# util.curve_to_csv("intermediate/ps_ella.csv", test_acc_el_ps, learned_size_el_ps)
# line_acc_el_ps, = plt.plot(learned_size_el_ps, test_acc_el_ps, label='Passive ELLA')
# handles.append(line_acc_el_ps)

# pool_dat2, init_dat2, test_dat2 = reload_dat()


# i1 = util.mat_eq(init_dat, init_dat2)
# i2 = util.mat_eq(test_dat, test_dat2)
# i3 = util.mat_eq(pool_dat, pool_dat2)

# # Clear Memory
# pool_dat, init_dat, test_dat = reload_dat()


# ### ELLA + Random Task Selection ###
# fig_flag[1] = True
# test_acc_el_rt, learned_size_el_rt = run_lm_trials(pool_dat, init_dat, test_dat, ELLA,
        # evaluation=config.EVAL_ME, engine=eng, do_active=False, task_ctrl=exp.TASK_RAND, n_trials=5)
# print test_acc_el_rt
# util.curve_to_csv("intermediate/rt_ella.csv", test_acc_el_rt, learned_size_el_rt)
# line_acc_el_rt, = plt.plot(learned_size_el_rt, test_acc_el_rt, label='Random Task')
# handles.append(line_acc_el_rt)


# print i1, i2, i3

# pool_dat = load_pool()
# init_dat = load_init()
# test_dat = load_test()
# gc.collect()

# pool_dat, init_dat, test_dat = reload_dat()

### ELLA + Active Task Selection ###
fig_flag[2] = True
test_acc_el_att, learned_size_el_att = run_lm_trials(pool_dat, init_dat, test_dat, ELLA,
        evaluation=config.EVAL_ME, engine=eng, do_active=False, task_ctrl=exp.TASK_ACTIVE, n_trials=5)
print test_acc_el_att
line_acc_el_att, = plt.plot(learned_size_el_att, test_acc_el_att, label='Active Task')
util.curve_to_csv("intermediate/att_ella.csv", test_acc_el_att, learned_size_el_att)
handles.append(line_acc_el_att)

# pool_dat = load_pool()
# init_dat = load_init()
# test_dat = load_test()
# # gc.collect()


### Active ELLA ###
fig_flag[3] = True
test_acc_el_al, learned_size_el_al = run_lm_trials(pool_dat, init_dat, test_dat, ELLA,
        evaluation=config.EVAL_ME, engine=eng, task_ctrl=exp.TASK_NONE, do_active=True, n_trials=5)
print test_acc_el_al
line_acc_el_al, = plt.plot(learned_size_el_al, test_acc_el_al, label='Active Learning')
util.curve_to_csv("res/al_ella.csv", test_acc_el_al, learned_size_el_al)
handles.append(line_acc_el_al)

# ### ELLA + Active Task Selection + Active Learning ###
# fig_flag[4] = True
# test_acc_el_act, learned_size_el_act = run_ml_landmine(pool_dat, init_dat, test_dat, ELLA, evaluation=config.EVAL_ME,
        # engine=eng, task_ctrl=exp.TASK_ACTIVE, do_active=True)
# print test_acc_el_act
# line_acc_el_act, = plt.plot(learned_size_el_act, test_acc_el_act, label='Active Task Selection + Active Learning')
# util.curve_to_csv("res/act_ella.csv", test_acc_el_act, learned_size_el_act)
# handles.append(line_acc_el_act)


# plt.legend(loc='lower right', handles=handles)
# plt.xlabel('Size of training set')
# plt.ylabel('Accuracy')
# util.save_fig('res/uncert_log_prob/', fig_cap, fig_flag, config.TRAIN_PERC, config.INS_SIZE,
        # config.N_ITER if config.ITER_ENABLE else -1, config.EVAL_ME)


print "Done"
# print test_acc_el_ps
# print learned_size_el_ps
# print test_acc_el_ac
# print learned_size_el_ac

# print "done"
