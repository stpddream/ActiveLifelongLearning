#!/usr/bin/env python
# encoding: utf-8
from data_process import gen_land_pool
import util
from config import T
import numpy as np
from query_strategy import model_uncert
import random
from learner import ELLA


class Professor:
    def __init__(self, init_dat, train_dat, multi_t=False, random=True, do_active=False, rand_task=False):
        self.trained_x = init_dat['feature']
        self.trained_y = init_dat['label']

        self.train_pool = gen_land_pool(train_dat, multi_t)
        self.pool_size = util.dat_size(train_dat)
        self.total_pool = self.pool_size
        self.init_size = util.dat_size(init_dat)

        self.multi_task = multi_t;
        self.rand_task = rand_task

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

    def get_trained_size(self):
        return self.init_size + self.total_pool - self.pool_size

    def has_next(self):
        return self.pool_size > 0

    def next_train_set(self, size, learner=None):

        # Order training set by informativeness in active learning
        if not self.has_next():
            return False

        if self.multi_task:

            print "len of tasks", len(self.task_online)
            t = int(learner.next_task(self.train_pool, rand_gen=self.rand_task)) - 1 # MINUS ONE BECAUSE MATLAB START WITH ONE!!
            print "next task", t

            # Handle if no training instance available for the task
            tr_size = min(size, self.train_pool[t].shape[0])

            # Active Instance Selection
            if self.do_active:
                comp_info_values(learner, self.train_pool[t], model_uncert, self.multi_task)
                # selected = self.train_pool[t][np.argpartition(-self.train_pool[t][:, -1], tr_size)[:tr_size], :]
                selected = self.train_pool[t][np.argsort(self.train_pool[t][:, -1])[::-1]]
            else : selected = self.train_pool[t][:tr_size, :]


            self.train_pool[t] = self.train_pool[t][tr_size:, :]

            # When task training set empty
            if not self.train_pool[t].shape[0]:
                print "task", t, "is empty"
                self.task_online.pop(t)
                self.train_pool = np.delete(self.train_pool, t, 0)


        else:
            if self.do_active:
                self.train_pool = comp_info_values(learner, self.train_pool, model_uncert, self.multi_task)

                # Optimize this
                self.train_pool = self.train_pool[np.argsort(self.train_pool[:, -1])[::-1]]

            tr_size = min(size, len(self.train_pool))
            selected = self.train_pool[:tr_size, :]
            self.train_pool = self.train_pool[tr_size:, :]

        self.pool_size -= tr_size

        # return x, y, task
        return selected[:,:-3], selected[:,-3], selected[:,-2]



def comp_info_values(learner, features, func, multi_t):
    """
    Compute information value based on coef
    """
    if multi_t:
        values = func(learner, features[:,:-3], features[0, -2])
        for idx, val in enumerate(values):
            features[idx, -1] = val

    else:
        if isinstance(learner, ELLA):
            # Grouping
            ins_groups = [[] for i in range(0, T)]

            for i in range(0, features.shape[0]):
                t = int(features[i, -2])
                ins_groups[t].append(features[i, :])

            for idx, task_grp in enumerate(ins_groups):
                if task_grp: # If list is not empty
                    values = func(learner, np.array(task_grp)[:, :-3], idx)

                    for r_idx, row in enumerate(task_grp):
                        row[-1] = values[r_idx]
        else:
            for row in features:
                t = int(row[-2])
                row[-1] = func(learner, row[:-3], t)

    return features

# Naive top k selection
def top_k(pool, k):
    top_pool = pool[0][np.argpartition(pool[0], k), :]
    for t in range(1, pool.shape[0]):
        top_pool = np.hstack((top_pool, pool[t][np.argpartition[pool[t], k], :]))

    return top_pool[np.argpartition[top_pool[t], k], :]
