#!/usr/bin/env python
# encoding: utf-8

from util import load_test
from util import load_pool
from util import load_init
import util

import matlab.engine

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
test_dat = load_test()
train_pool = load_pool()
init_dat = load_init()

init_lab = util.dat_to_list(init_dat)


## Init Data ##
### Reimplement ELLA in python?
## Pick

model = eng.init_model(init_lab)
print model




