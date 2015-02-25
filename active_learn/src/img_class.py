#!/usr/bin/env python
# encoding: utf-8

from sklearn.cross_validation import train_test_split
from sklearn import svm
import sklearn
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from load_smiles import *
from active_learner import *
import timeit
import util
from sklearn.datasets import load_digits

figure = plt.figure()

def pass_learn(x_train_pool, x_test, y_train_pool, y_test, train_perc, init_size, end_size, step):

    test_acc = []
    test_size = []

    n = 0
    for t_size in range(init_size, end_size, step):
        x_train = x_train_pool[:t_size]
        y_train = y_train_pool[:t_size]

        ####### SVM #######
        # svm_params = [{'kernel': ['linear'], 'C': [100]}]
        model = svm.SVC(kernel='linear', C=100)
        model.fit(x_train, np.reshape(y_train, y_train.shape[0]))
        test_acc.append(model.score(x_test, y_test))
        test_size.append(t_size)
        n += 1

    res = np.array((test_acc, test_size))
    train_line, = plt.plot(test_size, test_acc, label="Passive Learner")
    plt.xlabel('Size of training set')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1.0])
    print res

    print "average acc:", np.mean(test_acc)


def active_learn(x_train_pool, x_test, y_train_pool, y_test, train_perc, init_size, end_size, step):

    active_learner = USActiveLearner(svm.SVC(kernel='linear', C=100), x_train_pool, y_train_pool, init_size, step)
    test_size = []
    test_acc = []

    n = 1
    for t_size in range(init_size, end_size, step):
        if not active_learner.has_next():
            break

        active_learner.train_next_set()
        test_size.append(active_learner.get_trained_size())
        test_acc.append(active_learner.score(x_test, y_test))
        n += 1

    res = np.array((test_acc, test_size))
    print res

    train_line, = plt.plot(test_size, test_acc, label="Active Learner")



# DT_SIZE = 500
TRAIN_PERC = 50
START_SIZE = 10
END_SIZE = 10000
STEP = 50


# Load Data
print "Reading data..."

# data = util.load_mat("dat/hiva.mat", "dat/hiva.label")
data = load_smiles()

# Using the same train and test pool
x_train, x_test, y_train, y_test = train_test_split(data.data,
        data.target, train_size=TRAIN_PERC/100.0)

x_train, y_train = util.shuffle_ins(x_train, y_train)

print "train size", y_train.shape[0]
print "test size", y_test.shape[0]

####### Passive Learning ######
print("[Passive learning] ....\n")
s_time = timeit.default_timer()
pass_learn(x_train, x_test, y_train, y_test, TRAIN_PERC, START_SIZE, END_SIZE, STEP)
pass_time = timeit.default_timer() - s_time

# ###### Active Learning ######
# print ("[Active learning]")
# s_time = timeit.default_timer()
# active_learn(x_train, x_test, y_train, y_test, TRAIN_PERC, START_SIZE, END_SIZE, STEP)
# act_time = timeit.default_timer() - s_time

# print "Passive running time", pass_time
# print "Active running time", act_time

print "Producing legend"
plt.legend(loc='lower right')
print "Saving figure..."
plt.savefig("gf.jpg")




