#!/usr/bin/env python
# encoding: utf-8

from config import T
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import csv

def dat_size(dat):
    total = 0
    for t in range(0, T):
        total += dat['feature'][t].shape[0]
    return total

def load_pool():
    pool_f = open("data/pool", "rb")
    pool_dat = pk.load(pool_f)
    pool_f.close()
    print "pool size", dat_size(pool_dat)
    return pool_dat

def load_init():
    init_f = open("data/init", "rb")
    init_dat = pk.load(init_f)
    init_f.close()
    print "init size", dat_size(init_dat)
    return init_dat

def load_test():
    test_f = open("data/test", "rb")
    test_dat = pk.load(test_f)
    test_f.close()
    print "test size", dat_size(test_dat)
    return test_dat

def curve_add(acc, size):
    plt.plot(size, acc)

def learning_curve(file_name, acc, size):
    figure = plt.figure()
    plt.plot(size, acc)
    plt.savefig(file_name)

def curve_to_csv(file_name, acc, size):
    file = open(file_name, "wb")
    writer = csv.writer(file)
    writer.writerow(acc)
    writer.writerow(size)

def contain_one(labels):
    return np.bincount(labels).shape[0] != 1


def dat_to_mat(dat):
    for t in range(0, T):
        dat['feature'][t] = matlab.double(dat['feature'][t].tolist())
        lab_col = np.reshape(dat['label'][t], (dat['label'][t].shape[0], 1)) #Convert to column vector
        dat['label'][t] = matlab.double(lab_col.tolist())
    return dat

def pool2mat(dat):
    mat_x = []
    mat_y = []
    for t in range(0, dat.shape[0]):
        mat_x.append(matlab.double(dat[t][:, :-3].tolist()))
        mat_y.append(matlab.double(dat[t][:, -3].tolist()))

    return mat_x, mat_y



def trow2mat(task_rows):

    task_mat = []
    # if ys
    if len(task_rows[0].shape) < 2:
        for t in range(0, T):
            lab_col = np.reshape(task_rows[t], (task_rows[t].shape[0], 1)) #Convert to column vector
            task_mat.append(matlab.double(lab_col.tolist()))

    # if x
    else:
        for t in range(0, T):
            task_mat.append(matlab.double(task_rows[t].tolist()))

    return task_mat



def models_act(models):
    count = 0
    for t in range(0, T):
        if models[t].is_activated():
            count += 1
    return count


def add_bias(data):
    for t in range(0, T):
        bias_col = np.empty((data['feature'][t].shape[0], 1))
        bias_col.fill(1)
        data['feature'][t] = np.hstack((data['feature'][t], bias_col))
    return data

def arr2mat(nparray):
    return matlab.double(nparray.tolist())

# Just matlab vector
def matarr2list(matarr):
    print matarr
    list = []
    for ele in matarr:
        list.append(ele)
    return list


