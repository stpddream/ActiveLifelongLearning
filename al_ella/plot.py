__author__ = 'stpanda'

import matplotlib.pyplot as plt
import csv
import config

#### Process Non active learning
file_non = open('res/non_ac.csv', 'r')

non_ac_acc = []
non_ac_size = []

r_cnt = 0
for row in csv.reader(file_non):
    if r_cnt == 0:
        non_ac_acc = row
    else: non_ac_size = row
    r_cnt += 1


file_ac_multi = open('res/ac_multi.csv', 'r')

ac_multi_acc = []
ac_multi_size = []

r_cnt = 0
for row in csv.reader(file_ac_multi):
    if r_cnt == 0:
        ac_multi_acc = row
    else: ac_multi_size = row
    r_cnt += 1

print ac_multi_acc
print ac_multi_size

figure = plt.figure()
line_non_act, = plt.plot(non_ac_size, non_ac_acc, label='Non Active')
line_act, = plt.plot(ac_multi_size, ac_multi_acc, label='Active')
plt.legend(loc='lower right', handles=[line_non_act, line_act])
plt.xlabel('Size of training set')
plt.ylabel('Accuracy')
plt.savefig('res/uncert_log_prob/' + str(config.TRAIN_PERC) + '_' + str(config.INS_SIZE) + '.png')
plt.show()





