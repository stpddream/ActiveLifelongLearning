__author__ = 'stpanda'

import matplotlib.pyplot as plt


def read_dat_file(fl):
    data_non_str = fl.read().replace('\n', '').replace('[', '')
    non_dat = data_non_str.split(']')[:2]
    print non_dat
    return non_dat[1].strip().split(), non_dat[0].strip().split()

#### Process Non active learning
file_non = open('../res/svm_plain/svm_one.res', 'r')
non_x, non_y = read_dat_file(file_non)

#### Process Active Learning
file_act = open('../res/active/svm_one.res', 'r')
act_x, act_y = read_dat_file(file_act)

figure = plt.figure()
plt.plot(non_x, non_y)
plt.plot(act_x, act_y)
plt.xlabel('Size of training set')
plt.ylabel('Accuracy')
plt.show()


