from config import T
import numpy as np
import util
from util import arr2mat
from util import trow2mat
import random

from sklearn.metrics import roc_auc_score

def ella_score(ella_model, test_dat):
    total_score = 0.0
    for t in range(0, T):
        total_score += ella_model.score(test_dat['feature'][t], test_dat['label'][t], t)
    return total_score / T

def ella_auc_score(ella_model, test_dat):
    total_score = 0.0
    for t in range(0, T):
        preds = ella_model.predict_proba(test_dat['feature'][t], t)
        total_score += roc_auc_score(test_dat['label'][t], preds)

    return total_score / T

class ELLA:

    def __init__(self, engine, init_dat):
        self.engine = engine

        self.x_train = []
        self.y_train = []

        for t in range(0, T):
            self.x_train.append(init_dat['feature'][t])
            self.y_train.append(init_dat['label'][t])

        self.model_struct = engine.init_model(util.dat_to_mat(init_dat))


    def predict_proba(self, Xs, t):
        return self.engine.predictELLA(self.model_struct, arr2mat(Xs), t + 1)

    def score(self, Xs, Ys, t):

        # Matlab numbering starts from 1
        preds = self.engine.predictELLA(self.model_struct, arr2mat(Xs), t + 1)
        pred_lab = [1 if (pred[0] - 0.5) > 0 else 0 for pred in preds]
        res = [1 if (pred_lab[i] - Ys[i] == 0) else 0 for i in range(0, len(preds))]

        return float(sum(res)) / len(preds)

    def refine_model(self, Xs, Ys, t):
        # Update training instances
        self.x_train[t] = np.vstack((self.x_train[t], Xs))
        self.y_train[t] = np.concatenate((self.y_train[t], Ys))

        self.model_struct = self.engine.dropTaskELLA(self.model_struct, t + 1)
        self.model_struct = self.engine.addTaskELLA(self.model_struct, arr2mat(self.x_train[t]), arr2mat(self.y_train[t]), t + 1)

    # Not working!!
    def get_model(self, task):
        print self.model_struct.keys()
        print self.model_struct['L']
        print self.model_struct['S']
        print self.model_struct['S'][0][task]
        print self.model_struct['S'].size

        model_params = self.engine.get_model_param(self.model_struct['L'], self.model_struct['S'], task)
        print model_params
        print util.matarr2list(model_params)
        return util.matarr2list(model_params)

    def next_task(self, seed_dat, rand_gen=False):
        # turn seed data into matrix form (tasks sit next to each other)
        select_criterion = 2
        if rand_gen:
            select_criterion = 1

        mat_x, mat_y = util.pool2mat(seed_dat)
        task_id = self.engine.selectTaskELLA(self.model_struct, mat_x, mat_y, select_criterion)
        return task_id


