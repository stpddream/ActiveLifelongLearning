#!/usr/bin/env python
# encoding: utf-8

from experiment import run_ml_landmine
from learner import MLLearner
from learner import ELLA
from util import dat_size
from util import learning_curve
import matlab.engine

import matplotlib.pyplot as plt

import util
import config
import experiment as exp

from util import load_test
from util import load_init
from util import load_pool

pool_dat = load_pool()
init_dat = load_init()
test_dat = load_test()

#############################
###### STL Experiments ######
#############################

# print "[info]Start passive learning..."
# test_acc_ps, learned_size_ps = run_ml_landmine(pool_dat, init_dat, test_dat, MLLearner, evaluation=exp.EVAL_ACC, do_active=False)
# util.curve_to_csv("res/ps_stl_non.csv", test_acc_ps, learned_size_ps)

print "[info]Start active learning..."
test_acc_ac, learned_size_ac = run_ml_landmine(pool_dat, init_dat, test_dat, MLLearner, evaluation=exp.EVAL_ACC, do_active=True)
util.curve_to_csv("res/acc_stl_non.csv", test_acc_ac, learned_size_ac)

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

# # Set up environment
# print "Starting matlab..."
# eng = matlab.engine.start_matlab()

# ELLA_DIR = "/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/lib/ELLAv1.0"
# eng.addpath("/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/ml")
# eng.addpath(eng.genpath(ELLA_DIR))

# # run experiment
# test_acc_el_ps, learned_size_el_ps = run_ml_landmine(pool_dat, init_dat, test_dat, ELLA,
        # evaluation=exp.EVAL_ACC, engine=eng, do_active=False, rand_task=True)
# test_acc_el_att, learned_size_el_att = run_ml_landmine(pool_dat, init_dat, test_dat, ELLA,
        # evaluation=exp.EVAL_ACC, engine=eng, do_active=False, rand_task=False)

# util.curve_to_csv("res/ps_ella.csv", test_acc_el_ps, learned_size_el_ps)
# util.curve_to_csv("res/att_ella.csv", test_acc_el_att, learned_size_el_att)


# test_acc_el_ac, learned_size_el_ac = run_ml_landmine(pool_dat, init_dat, test_dat, ELLA, evaluation=exp.EVAL_ACC, engine=eng, do_active=True)

# figure = plt.figure()
# line_stl_ps, = plt.plot(learned_size_el_ps, test_acc_el_ps, label='Passive ELLA')
# line_stl_ac, = plt.plot(learned_size_el_att, test_acc_el_att, label='Active')
# plt.legend(loc='lower right', handles=[line_stl_ps, line_stl_ac])
# plt.xlabel('Size of training set')
# plt.ylabel('Accuracy')
# plt.savefig('res/uncert_log_prob/' + 'acc_' + str(config.TRAIN_PERC) + '_' + str(config.INS_SIZE) + '_el.png')

# print test_acc_el_ps
# print learned_size_el_ps
# print test_acc_el_ac
# print learned_size_el_ac

# print "done"
