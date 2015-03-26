__author__ = 'stpanda'

import matplotlib.pyplot as plt
import csv
import config
import util

def read_res(file_name):
    file = open(file_name, 'r')
    acc = []
    size = []
    r_cnt = 0
    for row in csv.reader(file):
        if r_cnt == 0:
            acc = row
        else: size = row
        r_cnt += 1
    return acc, size

figure = plt.figure()
handles = []


fig_label = ['STL-PS', 'STL-AT', 'ELLA', 'ELLA-RTS', 'ELLA-ATS', 'ELLA-AIS' , 'ELLA-ATAL']
# fig_cap = ['stlps', 'stlal', 'ellaps', 'ellart', 'ellaat', 'ellaal', 'ellaact']
fig_flag = [False for i in range(0, len(fig_label))]

# fig_flag[fig_label.index('STL-PS')] = True
# fig_flag[fig_label.index('STL-AT')] = True
fig_flag[fig_label.index('ELLA')] = True
fig_flag[fig_label.index('ELLA-RTS')] = True
fig_flag[fig_label.index('ELLA-ATS')] = True
fig_flag[fig_label.index('ELLA-AIS')] = True
# fig_flag[fig_label.index('ELLA-ATAL')] = True

it_count = sum(fig_flag)


# Process and generate plots
for idx in range(0, len(fig_label)):
    if fig_flag[idx]:
        acc, size = read_res('intermediate/' + fig_label + '.csv')
        line, = plt.plot(size, acc, label=fig_label[idx])
        handles.append(line)

# #### Process Non active learning
# file_non = open('res/non_ac.csv', 'r')
# r_cnt = 0

# #### Active STL ####
# file_ac_multi = open('res/ac_multi.csv', 'r')

# #### Passive ELLA ####
# fig_flag[0] = True
# acc_el_ps, size_el_ps = read_res('intermediate/ps_ella.csv')
# line_acc_el_ps, = plt.plot(size_el_ps, acc_el_ps, label='ELLA')
# handles.append(line_acc_el_ps)

# #### Random Task ELLA ####
# fig_flag[1] = True
# acc_el_rt, size_el_rt = read_res('intermediate/rt_ella.csv')
# line_acc_el_rt, = plt.plot(size_el_rt, acc_el_rt, label='Random Task')
# handles.append(line_acc_el_rt)

# ### ELLA + Active Task Selection ###
# fig_flag[2] = True
# acc_el_att, size_el_att = read_res('intermediate/att_ella.csv')
# line_acc_el_att, = plt.plot(size_el_att, acc_el_att, label='ELLA + Active Task')
# handles.append(line_acc_el_att)

# ### Active ELLA ####
# fig_flag[3] = True
# acc_el_al, size_el_al = read_res('intermediate/al_ella.csv')
# line_acc_el_al, = plt.plot(size_el_al, acc_el_al, label='ELLA + Active Instance')
# handles.append(line_acc_el_al)

#### ELLA + Active Task + Active Learning ####
# fig_flag[4] = True
# acc_el_act, size_el_act = read_csv('intermediate/act_ella.csv')
# line_acc_el_act, = plt.plot(size_el_act, acc_el_act, label='Active Task Selection + Active Learning')
# handles.append(line_acc_el_act)

plt.legend(loc='lower right', handles=handles)
plt.xlabel('Size of training set')
plt.ylabel('Accuracy')
util.save_fig('res/uncert_log_prob/', it_count, fig_label, fig_flag, config.TRAIN_PERC, config.INS_SIZE,
        config.N_ITER if config.ITER_ENABLE else -1, config.EVAL_ME)



