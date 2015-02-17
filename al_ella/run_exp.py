#!/usr/bin/env python
# encoding: utf-8

from proc_data import load_landmine
from proc_data import gen_land_pool
from proc_data import prod_init_dat
from learner import Learner

import numpy as np
from sklearn.linear_model import LogisticRegression
from config import T
from config import INS_SIZE
from active_learn import comp_info_values
from active_learn import model_uncert
from active_learn import model_score
from numpy.random import shuffle



######## Panda ######
# Comparing Multiple Active Learner vs ELLA + ATS vs ELLA + ATS + AL vs
# ELLA + AL
#####################



## Process Data ##
land_test = np.load("data/land_test")
land_train_pl = np.load("data/land_train_pl")
init_dat = np.load("data/land_init")
train_pool = np.array(gen_land_pool(land_train_pl))
shuffle(train_pool)

print len(init_dat['feature'][:])

## Train Initial Model ##
models = []

for t in range(0, T):
    while True:
        try:
            models.append(Learner(LogisticRegression(), init_dat['feature'][t],
                init_dat['label'][t]))
            break
        except ValueError:
            print "value error"
            continue


print "Start training..."

### Training Until No more data available ###
count = 20
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
    print model_score(models, land_test)
    if count < 0: break
    count -= 1

# for model in models:
    # print "size ", model.get_trained_size()







