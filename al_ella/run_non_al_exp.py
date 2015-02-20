#!/usr/bin/env python
# encoding: utf-8

from proc_data import gen_land_pool
from learner import Learner
from util import dat_size
from util import learning_curve

import pickle as pk
import numpy as np
from sklearn.linear_model import LogisticRegression
from config import T
from config import INS_SIZE
from active_learn import model_score
from numpy.random import shuffle




######## Panda ######
# Comparing Multiple Active Learner vs ELLA + ATS vs ELLA + ATS + AL vs
# ELLA + AL
# This file runs Pure Multiple Active Learner
#####################

###### Load Data ######
pool_f = open("data/pool", "rb")
pool_dat = pk.load(pool_f)
pool_f.close()
print "pool size", dat_size(pool_dat)

init_f = open("data/init", "rb")
init_dat = pk.load(init_f)
init_f.close()
print "init size", dat_size(init_dat)

test_f = open("data/test", "rb")
test_dat = pk.load(test_f)
test_f.close()
print "test size", dat_size(test_dat)

train_pool = np.array(gen_land_pool(pool_dat))
shuffle(train_pool)

###### Train Initial Model ######
models = []

for t in range(0, T):
    # while True:
        # try:
            # models.append(Learner(LogisticRegression(), init_dat['feature'][t],
                # init_dat['label'][t]))
            # break
        # except ValueError:
            # print "value error"
            # continue
    models.append(Learner(LogisticRegression(), init_dat['feature'][t],
                init_dat['label'][t]))


print "Start training..."


print "pool", len(train_pool)
total_pool_size = len(train_pool)
test_acc = []
learned_size= []

### Training Until No more data available ###
count = 1000
while train_pool.size:
    print "pool", len(train_pool)
    tr_size = min(INS_SIZE, len(train_pool))
    selected = train_pool[:INS_SIZE, :]

    next_train_x = [[] for i in range(0, T)]
    next_train_y = [[] for i in range(0, T)]

    for row in selected:
        t = int(row[10])
        next_train_x[t].append(row[:9])
        next_train_y[t].append(row[9])

    for t in range(0, T):
        if next_train_x[t]:
            models[t].refine_model(np.array(next_train_x[t]), np.array(next_train_y[t]))
    train_pool = train_pool[INS_SIZE:, :]

    acc =  model_score(models, test_dat)
    test_acc.append(acc)
    learned_size.append(total_pool_size - len(train_pool))

    # if count < 0: break
    # count -= 1

print test_acc
print learned_size
learning_curve("fig_learn.png", test_acc, learned_size)


# for model in models:
    # print "size ", model.get_trained_size()

