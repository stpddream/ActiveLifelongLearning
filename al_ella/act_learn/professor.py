#!/usr/bin/env python
# encoding: utf-8
from data_process import gen_land_pool
import util
from config import T
import numpy as np
from query_strategy import model_uncert


class Professor:
    def __init__(self, init_dat, train_dat, multi_t=False, random=True, do_active=False):
        self.trained_x = init_dat['feature']
        self.trained_y = init_dat['label']

        self.train_pool = gen_land_pool(train_dat, multi_t)
        self.pool_size = util.dat_size(train_dat)
        self.multi_task = multi_t;

        # Keep track of tasks that have training instances
        # In matlab format where all tasks start with one
        self.task_online = [t for t in range(0, T)]

        if random:
            if multi_t:
                for task_pool in self.train_pool:
                    np.random.shuffle(task_pool)
            else:
                np.random.shuffle(self.train_pool)

        self.do_active = do_active

    def get_pool_size(self):
        return self.pool_size

    def has_next(self):
        return self.pool_size > 0

    def next_train_set(self, size, learner=None):

        # Order training set by informativeness in active learning
        if self.do_active:
            train_pool = comp_info_values(learner, self.train_pool, model_uncert)
            self.train_pool = self.train_pool[np.argsort(train_pool[:, -1])[::-1]]

        if not self.has_next():
            return False

        if self.multi_task:

            print "len of tasks", len(self.task_online)
            t = int(learner.next_task(self.train_pool)) - 1 # MINUS ONE BECAUSE MATLAB START WITH ONE!!
            print "next task", t
            print self.train_pool.shape

            # Handle if no training instance available for the task
            tr_size = min(size, self.train_pool[t].shape[0])

            print self.train_pool[t].shape
            selected = self.train_pool[t][:tr_size, :]
            self.train_pool[t] = self.train_pool[t][tr_size:, :]

            # When task training set empty
            if not self.train_pool[t].shape[0]:
                print "task", t, "is empty"
                self.task_online.pop(t)
                self.train_pool = np.delete(self.train_pool, t, 0)

        else:
            tr_size = min(size, len(self.train_pool))
            selected = self.train_pool[:tr_size, :]
            self.train_pool = self.train_pool[tr_size:, :]

        self.pool_size -= tr_size

        # return x, y, task
        return selected[:,:-3], selected[:,-3], selected[:,-2]



def comp_info_values(learner, features, func):
    """
    Compute information value based on coef
    """

    for row in features:
        t = int(row[-2])
        row[-1] = func(row[:-3], learner.get_model(t))

    return features
