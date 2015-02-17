#!/usr/bin/env python
# encoding: utf-8

from config import T

def dat_size(dat):
    total = 0
    for t in range(0, T):
        total += dat['feature'][t].shape[0]
    return total


