#!/usr/bin/env python
# encoding: utf-8

import util
import matlab.engine
from experiment import run_lm_trials
from learner import ELLA
from learner import MLLearner
import experiment as exp
import config

import gc
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


# Set up environment
print "Starting matlab..."
eng = matlab.engine.start_matlab()

ELLA_DIR = "/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/lib/ELLAv1.0"
eng.addpath("/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/ml")
eng.addpath(eng.genpath(ELLA_DIR))

file_n = ['STL-PS', 'STL-AT', 'ELLA', 'ELLA-RTS', 'ELLA-ATS', 'ELLA-AIS' , 'ELLA-ATAL']

### Passive STL ###
# test_acc_ps, learned_size_ps = run_lm_trials(pool_dat, init_dat, test_dat, MLLearner,evaluation=config.EVAL_ME, do_active=False, n_trials=config.N_TRIAL)
# util.curve_to_csv("intermediate/" + file_n[0] + , test_acc_ps, learned_size_ps)

# ### Active STL ###
# test_acc_ac, learned_size_ac =run_lm_trials(pool_dat, init_dat, test_dat, MLLearner,
        # evaluation=config.EVAL_ME, do_active=True, n_trials=config.N_TRIAL)
# util.curve_to_csv("intermediate/" + file_n[1] + ".csv", test_acc_ac, learned_size_ac)


# # ### Passive ELLA ###
# test_acc_el_ps, learned_size_el_ps = run_lm_trials(pool_dat, init_dat, test_dat, ELLA,
        # evaluation=config.EVAL_ME, engine=eng, do_active=False, task_ctrl=exp.TASK_NONE, n_trials=config.N_TRIAL)
# util.curve_to_csv("intermediate/ps_ella.csv", test_acc_el_ps, learned_size_el_ps)


# ### Active ELLA ###
# test_acc_el_al, learned_size_el_al = run_lm_trials(pool_dat, init_dat, test_dat, ELLA,
        # evaluation=config.EVAL_ME, engine=eng, task_ctrl=exp.TASK_NONE, do_active=True, n_trials=config.N_TRIAL)
# util.curve_to_csv("intermediate/al_ella.csv", test_acc_el_al, learned_size_el_al)

# ### ELLA + Random Task Selection ###
# test_acc_el_rt, learned_size_el_rt = run_lm_trials(pool_dat, init_dat, test_dat, ELLA,
        # evaluation=config.EVAL_ME, engine=eng, do_active=False, task_ctrl=exp.TASK_RAND, n_trials=config.N_TRIAL)
# util.curve_to_csv("intermediate/rt_ella.csv", test_acc_el_rt, learned_size_el_rt)

## Active Task Selection ###
test_acc_el_att, learned_size_el_att = run_lm_trials(pool_dat, init_dat, test_dat, ELLA,
        evaluation=config.EVAL_ME, engine=eng, do_active=False, task_ctrl=exp.TASK_ACTIVE, n_trials=config.N_TRIAL)
util.curve_to_csv('intermediate/' + file_n[5] + '.csv', test_acc_el_att, learned_size_el_att)


# ### ELLA + Active Task Selection + Active Learning ###
# test_acc_el_act, learned_size_el_act = run_ml_landmine(pool_dat, init_dat, test_dat, ELLA, evaluation=config.EVAL_ME,
        # engine=eng, task_ctrl=exp.TASK_ACTIVE, do_active=True)
# util.curve_to_csv("intermediate/act_ella.csv", test_acc_el_act, learned_size_el_act)


print "Done"
