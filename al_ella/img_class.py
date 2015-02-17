#!/usr/bin/env python
# encoding: utf-8
from sklearn.cross_validation import train_test_split
from sklearn import svm
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from load_smiles import *
from active_learner import *
import timeit

figure = plt.figure()

def pass_learn(x_train_pool, x_test, y_train_pool, y_test, train_perc, start_perc, end_perc, step):
    test_acc = np.zeros((end_perc - start_perc) / step + 1)
    test_size = np.zeros((end_perc - start_perc) / step + 1)
    print len(test_acc)

    n = 0
    for percent in range(start_perc, end_perc + 1, step):
        t_size = int(percent * 0.01 * len(x_train_pool))
        print "size is ", len(x_train_pool)
        x_train = x_train_pool[:t_size]
        y_train = y_train_pool[:t_size]

        print np.array(y_train).shape
        print x_train.shape

        print "Training size:", t_size
        ####### SVM #######
        #svm_params = [{'kernel': ['rbf'], 'gamma':[1e-3, 1e-4], 'C':[1, 10, 100]}]
        svm_params = [{'kernel': ['linear'], 'C': [100]}]
        model = svm.SVC(kernel='linear', C=100)
        print "which line"
        model.fit(x_train, np.array(y_train))
        test_acc[n] = model.score(x_test, y_test)
        test_size[n] = t_size
        print "accuracy", test_acc[n]
        n += 1

    res = np.array((test_acc, test_size))
    train_line, = plt.plot(test_size, test_acc, label="Passive Learner")
    plt.xlabel('Size of training set')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1.0])

    print "average acc:", np.mean(test_acc)


def active_learn(x_train_pool, x_test, y_train_pool, y_test, train_perc, start_perc, end_perc, step):
    test_acc = np.zeros((end_perc - start_perc) / step + 1)
    test_size = np.zeros((end_perc - start_perc) / step + 1)

    active_learner = USActiveLearner(svm.SVC(kernel='linear', C=100), x_train_pool, y_train_pool, start_perc, step)
    test_acc[0] = active_learner.score(x_test, y_test)
    test_size[0] = active_learner.get_trained_size()

    print "initial test size", test_size[0]
    print "initial test accuracy", test_acc[0]

    n = 1
    while active_learner.has_next():
        print "------- -------"
        print "next set ===> ", n
        t_size = active_learner.train_next_set()
        test_size[n] = active_learner.get_trained_size()
        print "test size", test_size[n]
        test_acc[n] = active_learner.score(x_test, y_test)
        print "accuracy", test_acc[n]
        n += 1

    res = np.array((test_acc, test_size))
    print res

    train_line, = plt.plot(test_size, test_acc, label="Active Learner")



DT_SIZE = 500
TRAIN_PERC = 90
START_PERC = 20
END_PERC = 100
STEP = 1


# Load Data
print "Reading data..."
smile_data = load_smiles()
print "shape for data", smile_data.target.shape

# Using the same train and test pool
x_train_pool, x_test, y_train_pool, y_test = train_test_split(smile_data.data[:DT_SIZE],
        smile_data.target[:DT_SIZE], train_size=TRAIN_PERC/100.0)

print "x_train", x_train_pool

####### Passive Learning ######
print("passive learning....\n")
s_time = timeit.default_timer()
pass_learn(x_train_pool, x_test, y_train_pool, y_test, TRAIN_PERC, START_PERC, END_PERC, STEP)
pass_time = timeit.default_timer() - s_time

# ###### Active Learning ######
# print ("now active learning\n")
# s_time = timeit.default_timer()
# active_learn(x_train_pool, x_test, y_train_pool, y_test, TRAIN_PERC, START_PERC, END_PERC, STEP)
# act_time = timeit.default_timer() - s_time

# print "Passive running time", pass_time
# print "Active running time", act_time

# print "Producing legend"
# plt.legend(loc='upper right')
# print "Saving figure..."
# plt.savefig("gf.jpg")




