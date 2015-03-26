#!/usr/bin/env python
# encoding: utf-8
import matlab.engine
import gc
from data_process import load_init
from data_process import load_pool
from data_process import load_test

def reload_dat():
    gc.collect()
    pool_dat = load_pool()
    init_dat = load_init()
    test_dat = load_test()
    return pool_dat, init_dat, test_dat

# Set up environment
print "Starting matlab..."
eng = matlab.engine.start_matlab()

ELLA_DIR = "/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/lib/ELLAv1.0"
eng.addpath("/home/stpanda/Dropbox/STDreamSoft/Academics/SeniorThesis/Projects/al_ella/ml")
eng.addpath(eng.genpath(ELLA_DIR))

pool_dat, init_dat, test_dat = reload_dat()
