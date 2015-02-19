#!/usr/bin/env python
# encoding: utf-8

from config import T
import pickle as pk

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

def load_test():
    init_f = open("data/init", "rb")
    init_dat = pk.load(init_f)
    init_f.close()
    print "init size", dat_size(init_dat)

def load_init():
    test_f = open("data/test", "rb")
    test_dat = pk.load(test_f)
    test_f.close()
    print "test size", dat_size(test_dat)
