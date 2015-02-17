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

from random import shuffle



######## Panda ######
# Comparing Multiple Active Learner vs ELLA + ATS vs ELLA + ATS + AL vs
# ELLA + AL
#####################


land_train, land_test = load_landmine()
train_pl, init_dat = prod_init_dat(land_train, 100)
train_pool = np.array(gen_land_pool(train_pl))

print np.array(init_dat['feature'][0]).shape

print np.reshape(init_dat['label'][0], init_dat['label'][0].shape[0]).shape

shuffle(train_pool)

models = []

for t in range(0, T):
    while True:
        try:
            models.append(Learner(LogisticRegression(), init_dat['feature'][t],
                init_dat['label'][t]))
            break
        except ValueError:
            continue

# for t in range(0, T):
    # print "model ", t, models[t].get_trained_size()

print "Start training..."
print "After..."
# print len(train_pool)
# print values.shape

while train_pool.size:
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

    print "start training..."
    for t in range(0, T):
        if next_train_x[t]:
            print "t is", t
            print next_train_x[t]
            models[t].refine_model(np.array(next_train_x[t]), np.array(next_train_y[t]))

    train_pool = sorted_dat[INS_SIZE:, :]

    break

for model in models:
    print "size ", model.get_trained_size()







