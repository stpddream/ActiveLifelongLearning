#!/usr/bin/env python
# encoding: utf-8

from config import T
from sklearn.metrics import roc_auc_score
from ella_learner import ELLA


def model_roc_score(learner, test_dat):
    total_auc_score = 0.0

    if isinstance(learner, ELLA):
        for t in range(0, T):
            preds = learner.predict_proba(test_dat['feature'][t], t)
            total_auc_score += roc_auc_score(test_dat['label'][t], preds)
    else:
        for t in range(0, T):
            pred_proba = learner.predict_proba(test_dat['feature'][t], t)
            total_auc_score += roc_auc_score(test_dat['label'][t], pred_proba[:, 1])

    return total_auc_score / T


def model_score(learner, tests):
    total_score = 0.0

    for t in range(0, T):
        this_score = learner.score(tests['feature'][t], tests['label'][t], t)
        total_score += this_score

    return total_score / T
